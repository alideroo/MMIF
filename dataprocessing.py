import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread

def parse_args():
    parser = argparse.ArgumentParser(description='Create dataset')
    
    parser.add_argument('--data_root', type=str, default='train_img', 
                       help='Root directory containing training images')
    parser.add_argument('--ir_folder', type=str, default='train/ir',
                       help='Relative path to IR images folder')
    parser.add_argument('--vi_folder', type=str, default='train/vi',
                       help='Relative path to VI images folder')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for HDF5 file')
    
    parser.add_argument('--img_size', type=int, default=128,
                       help='Patch size (height and width)')
    parser.add_argument('--stride', type=int, default=200,
                       help='Stride for patch extraction')
    
    parser.add_argument('--fraction_threshold', type=float, default=0.1,
                       help='Contrast threshold for filtering low contrast patches')
    parser.add_argument('--lower_percentile', type=int, default=10,
                       help='Lower percentile for contrast calculation')
    parser.add_argument('--upper_percentile', type=int, default=90,
                       help='Upper percentile for contrast calculation')
    
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Custom dataset name')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing HDF5 file')
    
    parser.add_argument('--max_patches_per_image', type=int, default=None,
                       help='Maximum number of patches to extract per image')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def get_img_file(file_name):
    imagelist = []
    supported_formats = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')
    
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(supported_formats):
                imagelist.append(os.path.join(parent, filename))
    return imagelist

def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10, upper_percentile=90):
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    if limits[1] == 0: 
        return True
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

def validate_paths(args):
    ir_path = os.path.join(args.data_root, args.ir_folder)
    vi_path = os.path.join(args.data_root, args.vi_folder)
    
    if not os.path.exists(ir_path):
        raise FileNotFoundError(f"IR folder not found: {ir_path}")
    if not os.path.exists(vi_path):
        raise FileNotFoundError(f"VI folder not found: {vi_path}")
    
    return ir_path, vi_path

def create_output_filename(args):
    dataset_name = args.dataset_name if args.dataset_name else args.data_root
    filename = f"{dataset_name}_imgsize_{args.img_size}_stride_{args.stride}.h5"
    return os.path.join(args.output_dir, filename)

def process_image_pair(ir_file, vi_file, args):
    try:
        I_VI = imread(vi_file).astype(np.float32)
        I_IR = imread(ir_file).astype(np.float32)
        
        if len(I_VI.shape) == 3:
            I_VI = I_VI.transpose(2, 0, 1) / 255.
            I_VI = rgb2y(I_VI)
        else: 
            I_VI = I_VI[None, :, :] / 255. 
            
        if len(I_IR.shape) == 3: 
            I_IR = np.mean(I_IR, axis=2) 
        I_IR = I_IR[None, :, :] / 255.
        
        I_IR_Patch_Group = Im2Patch(I_IR, args.img_size, args.stride)
        I_VI_Patch_Group = Im2Patch(I_VI, args.img_size, args.stride)
        
        valid_patches = []
        total_patches = I_IR_Patch_Group.shape[-1]
        max_patches = args.max_patches_per_image if args.max_patches_per_image else total_patches
        
        for ii in range(min(total_patches, max_patches)):
            bad_IR = is_low_contrast(I_IR_Patch_Group[0, :, :, ii], 
                                   args.fraction_threshold, 
                                   args.lower_percentile, 
                                   args.upper_percentile)
            bad_VI = is_low_contrast(I_VI_Patch_Group[0, :, :, ii],
                                   args.fraction_threshold,
                                   args.lower_percentile,
                                   args.upper_percentile)
            
            if not (bad_IR or bad_VI):
                avl_IR = I_IR_Patch_Group[0, :, :, ii][None, ...]
                avl_VI = I_VI_Patch_Group[0, :, :, ii][None, ...]
                valid_patches.append((avl_IR, avl_VI))
        
        return valid_patches
        
    except Exception as e:
        print(f"Error processing {ir_file} and {vi_file}: {e}")
        return []

def main():
    args = parse_args()
    
    print(f"Data root: {args.data_root}")
    print("-" * 50)

    ir_path, vi_path = validate_paths(args)
    IR_files = sorted(get_img_file(ir_path))
    VI_files = sorted(get_img_file(vi_path))
    
    if len(IR_files) != len(VI_files):
        raise ValueError(f"Number of IR images ({len(IR_files)}) != number of VI images ({len(VI_files)})")
    
    if len(IR_files) == 0:
        raise ValueError("No images found in the specified directories")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_file = create_output_filename(args)
    
    if os.path.exists(output_file) and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}. Use --overwrite to overwrite.")
    
    with h5py.File(output_file, 'w') as h5f:
        h5_ir = h5f.create_group('ir_patchs')
        h5_vi = h5f.create_group('vi_patchs')
        
        train_num = 0
        total_processed = 0
        
        for i in tqdm(range(len(IR_files)), desc="Processing images"):
            if args.verbose:
                print(f"\\nProcessing: {os.path.basename(IR_files[i])} and {os.path.basename(VI_files[i])}")
            
            valid_patches = process_image_pair(IR_files[i], VI_files[i], args)
            
            for avl_IR, avl_VI in valid_patches:
                h5_ir.create_dataset(str(train_num), data=avl_IR, 
                                   dtype=avl_IR.dtype, shape=avl_IR.shape)
                h5_vi.create_dataset(str(train_num), data=avl_VI, 
                                   dtype=avl_VI.dtype, shape=avl_VI.shape)
                train_num += 1
            
            total_processed += 1



if __name__ == "__main__":
    main()