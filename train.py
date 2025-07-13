import argparse
from model.net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.dataset import H5Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia
from model.ae import NeuralSampler, Log_distance_coef, gradient_penalty_one_centered, Discriminator
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Image Fusion Training')
    
    parser.add_argument('--num_epochs', type=int, help='Total number of epochs')
    parser.add_argument('--epoch_gap', type=int, help='Epochs of Phase I')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--gpu_id', type=str, default='3', help='GPU device ID')
    
    parser.add_argument('--coeff_mse_loss_ri', type=float, default=1.0, help='Coefficient for RI MSE loss (alpha1)')
    parser.add_argument('--coeff_mse_loss_pet', type=float, default=1.0, help='Coefficient for PET MSE loss')
    parser.add_argument('--coeff_decomp', type=float, default=2.0, help='Coefficient for decomposition loss (alpha2 and alpha4)')
    parser.add_argument('--coeff_tv', type=float, default=5.0, help='Coefficient for TV loss')
    parser.add_argument('--coeff_d', type=float, default=0.01, help='Coefficient for discriminator loss')
    
    parser.add_argument('--clip_grad_norm_value', type=float, default=0.01, help='Gradient clipping value')
    parser.add_argument('--optim_step', type=int, default=20, help='Scheduler step size')
    parser.add_argument('--optim_gamma', type=float, default=0.5, help='Scheduler gamma')
    
    parser.add_argument('--data_path', type=str, default='data/train_img_imgsize_128_stride_200.h5', 
                       help='Path to training data')
    parser.add_argument('--save_dir', type=str, default='models/', help='Directory to save models')
    
    parser.add_argument('--base_dim', type=int, default=64, help='Base feature dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--detail_num_layers', type=int, default=1, help='Number of detail layers')
    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimension')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    criteria_fusion = Fusionloss()
    
    device = 'cuda'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=args.base_dim, num_heads=args.num_heads)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=args.detail_num_layers)).to(device)
    
    optimizer1 = torch.optim.Adam(Encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(Decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=args.optim_step, gamma=args.optim_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.optim_step, gamma=args.optim_gamma)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=args.optim_step, gamma=args.optim_gamma)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=args.optim_step, gamma=args.optim_gamma)
    
    MSELoss = nn.MSELoss()  
    L1Loss = nn.L1Loss()
    Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
    
    trainloader = DataLoader(H5Dataset(args.data_path),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0)
    
    loader = {'train': trainloader}
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    
    sample_s = nn.DataParallel(NeuralSampler(z_dim=args.z_dim)).to(device)
    log_distance_coef = nn.DataParallel(Log_distance_coef()).to(device)
    disc_block = nn.DataParallel(Discriminator(args.z_dim, args.z_dim)).to(device)
    optimizerDIS = torch.optim.Adam(list(disc_block.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    torch.backends.cudnn.benchmark = True
    prev_time = time.time()
    
    print(f"Training Configuration:")
    print(f"Epochs: {args.num_epochs}, Phase I epochs: {args.epoch_gap}")
    print(f"Learning rate: {args.lr}, Batch size: {args.batch_size}")
    print("-" * 50)
    
    for epoch in range(args.num_epochs):
        for i, (data_VI, data_IR) in enumerate(loader['train']):
            data_VI, data_IR = data_VI.cuda(), data_IR.cuda()
            Encoder.train()
            Decoder.train()
            BaseFuseLayer.train()
            DetailFuseLayer.train()
            
            Encoder.zero_grad()
            Decoder.zero_grad()
            BaseFuseLayer.zero_grad()
            DetailFuseLayer.zero_grad()
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            
            if epoch < args.epoch_gap:  # Phase I
                feature_V_B, feature_V_D, _ = Encoder(data_VI)
                feature_I_B, feature_I_D, _ = Encoder(data_IR)
                data_VI_hat, _, _1 = Decoder(data_VI, feature_V_B, feature_V_D)
                data_IR_hat, _, _1 = Decoder(data_IR, feature_I_B, feature_I_D)
                
                zs1, zs2 = sample_s(args.batch_size)
                zs = Decoder.module.fusion_module(zs1, zs2)
                x2s, _, sigmas2 = Decoder(None, zs1, zs2)
                
                scale_z = zs.std(dim=0).mean()
                scale_z = torch.clamp(scale_z, min=1e-8)
                scale_zx = x2s.std(dim=0).mean()
                scale_zx = torch.clamp(scale_zx, min=1e-8)
                
                dx_elem = (x2s[:,None,...] - x2s[None,:,...]).view(args.batch_size, args.batch_size, -1) * (log_distance_coef().exp() / scale_zx)
                dz_elem = (zs[:,None,:] - zs[None,:,:]).view(args.batch_size, args.batch_size, -1) * (log_distance_coef().exp() / scale_z)
                
                dx = dx_elem.norm(p=6, dim=2)
                dz = dz_elem.norm(p=6, dim=2)
                loss_gw = (torch.abs(dz - dx)).mean()
                
                logit_autoencoding = disc_block(data_VI, Decoder.module.fusion_module(feature_V_B, feature_V_D))
                logit_sampling = disc_block(x2s, zs)
                loss_d = (logit_sampling - logit_autoencoding).mean()
                
                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                mse_loss_V = 5 * Loss_ssim(data_VI, data_VI_hat) + MSELoss(data_VI, data_VI_hat)
                mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)
                
                Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VI),
                                     kornia.filters.SpatialGradient()(data_VI_hat))
                
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
                
                loss = (args.coeff_mse_loss_ri * mse_loss_V + 
                       args.coeff_mse_loss_pet * mse_loss_I + 
                       args.coeff_decomp * loss_decomp + 
                       args.coeff_tv * Gradient_loss + 
                       loss_gw + loss_d * args.coeff_d)
                
                loss.backward()
                nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm=args.clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm=args.clip_grad_norm_value, norm_type=2)
                optimizer1.step()
                optimizer2.step()
                
                feature_V_B, feature_V_D, _ = Encoder(data_VI)
                for _ in range(3):
                    data_VI_hat, _, _1 = Decoder(data_VI, feature_V_B, feature_V_D)
                    zs1, zs2 = sample_s(args.batch_size)
                    zs = Decoder.module.fusion_module(zs1, zs2)
                    x2s, _, sigmas2 = Decoder(None, zs1, zs2)
                    logit_autoencoding = disc_block(data_VI, Decoder.module.fusion_module(feature_V_B, feature_V_D))
                    logit_sampling = disc_block(x2s, zs)
                    loss_logits = -(logit_sampling - logit_autoencoding).mean()
                    loss_gp = (gradient_penalty_one_centered(data_VI, Decoder.module.fusion_module(feature_V_B, feature_V_D), x2s, zs, disc_block) +
                              1e-4 * logit_autoencoding.square().mean() +
                              1e-4 * logit_sampling.square().mean())
                    
                    loss_disc = loss_logits + loss_gp * 1
                    optimizerDIS.zero_grad()
                    loss_disc.backward(retain_graph=True)
                    optimizerDIS.step()
            
            else:  # Phase II
                feature_V_B, feature_V_D, feature_V = Encoder(data_VI)
                feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
                feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
                feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
                data_Fuse, feature_F, _ = Decoder(data_VI, feature_F_B, feature_F_D)
                
                batch_size_c, channels_c, height_c, width_c = feature_F_B.shape
                feature_F_B_flat = feature_F_B.view(args.batch_size, -1)
                feature_F_D_flat = feature_F_D.view(args.batch_size, -1)
                
                logits = 1 * F.normalize(feature_F_B_flat, dim=1) @ F.normalize(feature_F_D_flat, dim=1).T
                labels = torch.arange(logits.shape[0]).to(data_VI.device)
                accuracy = (logits.argmax(dim=1) == labels).float().mean()
                CL_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                
                mse_loss_V = 5 * Loss_ssim(data_VI, data_Fuse) + MSELoss(data_VI, data_Fuse)
                mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)
                
                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
                fusionloss, _, _ = criteria_fusion(data_VI, data_IR, data_Fuse)
                
                loss = fusionloss + args.coeff_decomp * loss_decomp + CL_loss
                loss.backward()
                
                nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm=args.clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm=args.clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=args.clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=args.clip_grad_norm_value, norm_type=2)
                
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()
            
            batches_done = epoch * len(loader['train']) + i
            batches_left = args.num_epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
                % (epoch, args.num_epochs, i, len(loader['train']), loss.item(), time_left)
            )
        
        scheduler1.step()
        scheduler2.step()
        if not epoch < args.epoch_gap:
            scheduler3.step()
            scheduler4.step()
        
        for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4]:
            if optimizer.param_groups[0]['lr'] <= 1e-6:
                optimizer.param_groups[0]['lr'] = 1e-6
    
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint = {
        'Encoder': Encoder.state_dict(),
        'Decoder': Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(args.save_dir, f"{timestamp}.pth"))


if __name__ == "__main__":
    main()