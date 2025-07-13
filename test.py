import argparse
from model.net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
import warnings
import logging
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import cv2

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    
    parser.add_argument('--ckpt_path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--dataset_name', type=str, required=True,
                       choices=['MRI_CT', 'MRI_PET', 'MRI_SPECT'],
                       help='Dataset name for testing')
    
    parser.add_argument('--base_dim', type=int, default=64, 
                       help='Base feature dimension')
    parser.add_argument('--num_heads', type=int, default=8, 
                       help='Number of attention heads')
    parser.add_argument('--detail_num_layers', type=int, default=1, 
                       help='Number of detail layers')

    parser.add_argument('--test_folder', type=str, default='test_img',
                       help='Root folder containing test images')
    parser.add_argument('--output_folder', type=str, default='test_result',
                       help='Root folder for saving test results')
    parser.add_argument('--gpu_id', type=str, default='2',
                       help='GPU device ID')
    
    parser.add_argument('--encoder_key', type=str, default='Encoder',
                       help='Key name for encoder in checkpoint')
    parser.add_argument('--decoder_key', type=str, default='Decoder',
                       help='Key name for decoder in checkpoint')
    parser.add_argument('--base_fuse_key', type=str, default='BaseFuseLayer',
                       help='Key name for base fusion layer in checkpoint')
    parser.add_argument('--detail_fuse_key', type=str, default='DetailFuseLayer',
                       help='Key name for detail fusion layer in checkpoint')
    
    parser.add_argument('--fusion_mode', type=str, default='standard',
                       choices=['standard', 'features_only'],
                       help='Fusion mode: standard or features_only')
    
    return parser.parse_args()

def load_model(args):

    device = 'cuda'
    
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(
        dim=args.base_dim, num_heads=args.num_heads)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(
        num_layers=args.detail_num_layers)).to(device)
    
    checkpoint = torch.load(args.ckpt_path)
    
    try:
        Encoder.load_state_dict(checkpoint[args.encoder_key])
        Decoder.load_state_dict(checkpoint[args.decoder_key])
        BaseFuseLayer.load_state_dict(checkpoint[args.base_fuse_key])
        DetailFuseLayer.load_state_dict(checkpoint[args.detail_fuse_key])
    except KeyError as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
        raise
    
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    
    return Encoder, Decoder, BaseFuseLayer, DetailFuseLayer, device

def process_images(args, Encoder, Decoder, BaseFuseLayer, DetailFuseLayer, device):

    test_folder_path = os.path.join(args.test_folder, args.dataset_name)
    output_folder_path = os.path.join(args.output_folder, args.dataset_name)
    
    os.makedirs(output_folder_path, exist_ok=True)
    
    modality1, modality2 = args.dataset_name.split('_')
    modality1_folder = os.path.join(test_folder_path, modality1)
    modality2_folder = os.path.join(test_folder_path, modality2)
    
    image_names = os.listdir(modality1_folder)
    
    with torch.no_grad():
        for img_name in image_names:
            
            data_IR = image_read_cv2(os.path.join(modality1_folder, img_name), mode='GRAY')
            data_IR = data_IR[np.newaxis, np.newaxis, ...] / 255.0
            
            data_VI_BGR = cv2.imread(os.path.join(modality2_folder, img_name))
            data_VI = cv2.split(cv2.cvtColor(data_VI_BGR, cv2.COLOR_BGR2YCrCb))[0]
            data_VI = data_VI[np.newaxis, np.newaxis, ...] / 255.0
            
            data_IR = torch.FloatTensor(data_IR).to(device)
            data_VI = torch.FloatTensor(data_VI).to(device)
            
            feature_V_B, feature_V_D, feature_V = Encoder(data_VI)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            
            if args.fusion_mode == 'standard':
                data_Fuse, _, _ = Decoder(data_IR + data_VI, feature_F_B, feature_F_D)
            else:
                data_Fuse, _, _ = Decoder(None, feature_F_B, feature_F_D)
            
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy()).astype(np.uint8)
            
            img_save(fi, img_name.split('.')[0], output_folder_path)

def evaluate_results(args):

    eval_folder = os.path.join(args.output_folder, args.dataset_name)
    ori_img_folder = os.path.join(args.test_folder, args.dataset_name)
    
    modality1, modality2 = args.dataset_name.split('_')
    modality1_folder = os.path.join(ori_img_folder, modality1)
    modality2_folder = os.path.join(ori_img_folder, modality2)
    
    metrics_all = {
        "EN": [], "SD": [], "SF": [], "MI": [], "SCD": [], "VIF": [], "Qabf": [], "SSIM": []
    }
    
    image_names = os.listdir(modality1_folder)
    
    for img_name in image_names:
        ir = image_read_cv2(os.path.join(modality1_folder, img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(modality2_folder, img_name), 'GRAY')
        fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), 'GRAY')
        
        metrics = np.array([
            Evaluator.EN(fi), 
            Evaluator.SD(fi), 
            Evaluator.SF(fi), 
            Evaluator.MI(fi, ir, vi),
            Evaluator.SCD(fi, ir, vi), 
            Evaluator.VIFF(fi, ir, vi), 
            Evaluator.Qabf(fi, ir, vi),
            Evaluator.SSIM(fi, ir, vi)
        ])
        
        metrics_all["EN"].append(metrics[0])
        metrics_all["SD"].append(metrics[1])
        metrics_all["SF"].append(metrics[2])
        metrics_all["MI"].append(metrics[3])
        metrics_all["SCD"].append(metrics[4])
        metrics_all["VIF"].append(metrics[5])
        metrics_all["Qabf"].append(metrics[6])
        metrics_all["SSIM"].append(metrics[7])
    
    model_name = os.path.basename(args.ckpt_path).split('.')[0]
    
    metric_averages = np.array([np.mean(metrics_all[metric]) for metric in metrics_all])
    
    return metric_averages, model_name

def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    print("="*80)
    print(f"Testing {args.dataset_name} with model: {os.path.basename(args.ckpt_path)}")
    print("="*80)
    
    print("Loading model...")
    Encoder, Decoder, BaseFuseLayer, DetailFuseLayer, device = load_model(args)
    print("Model loaded successfully...")
    
    process_images(args, Encoder, Decoder, BaseFuseLayer, DetailFuseLayer, device)
    
    print("Evaluating results...")
    metric_averages, model_name = evaluate_results(args)
    
    print("\n" + "="*80)
    print(f"Test Results for {args.dataset_name}:")
    print(f"Model: {model_name}")
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(f"{model_name}\t{metric_averages[0]:.3f}\t{metric_averages[1]:.3f}\t"
          f"{metric_averages[2]:.3f}\t{metric_averages[3]:.3f}\t{metric_averages[4]:.3f}\t"
          f"{metric_averages[5]:.3f}\t{metric_averages[6]:.3f}\t{metric_averages[7]:.3f}")
    
    summary_results = {
        'Model': [model_name],
        'Dataset': [args.dataset_name],
        'EN': [f"{metric_averages[0]:.3f}"],
        'SD': [f"{metric_averages[1]:.3f}"],
        'SF': [f"{metric_averages[2]:.3f}"],
        'MI': [f"{metric_averages[3]:.3f}"],
        'SCD': [f"{metric_averages[4]:.3f}"],
        'VIF': [f"{metric_averages[5]:.3f}"],
        'Qabf': [f"{metric_averages[6]:.3f}"],
        'SSIM': [f"{metric_averages[7]:.3f}"]
    }
    
    summary_df = pd.DataFrame(summary_results)
    summary_filename = os.path.join(args.output_folder, args.dataset_name, f"{model_name}_{args.dataset_name}_summary.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary results saved to: {summary_filename}")
    print("="*80)

if __name__ == "__main__":
    main()