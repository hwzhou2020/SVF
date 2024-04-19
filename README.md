# SVF
Single-shot volumetric flurorescence imaging with neural representations

## Data source
Please download data and put them in the "data" folder
Link:

## Scripts
Example bash script to run the fluo_recon.py code
```
CUDA_VISIBLE_DEVICES=0 python fluo_recon.py \
    --data_name AVG_roots_xyzScan_z2_128.tif \
    --exp_psf_name ExpPSF_605_20240311_shift.mat \
    --show_inter_imgs False \
    --model_opt "complie" \
    --if_log True \
    --wavelength 605e-9 \
    --lr_psf 5e-3 \
    --init_epochs 100 \
    --learn_psf_epochs 100 \
    --z_dim 8 \
    --z_sep 0.1 \
    --l1_g 0 \
    --l1_z 0
```

## Docker environment
Docker Hub ID
```
hwzhou/inr-repo:3d-pfm
```

## Dependent packages
If you run the code on Linux with pyTorch version >= 2.0.1, you can use ```--model_opt "compile"```. Otherwise, please us ```--model_opt "jit"```
```
argparse
matplotlib
numpy
os
scipy
skimage
sys
torch
tqdm
```

## Reference / Citation







