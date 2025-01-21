# Single-shot volumetric fluorescence imaging with neural fields

Authors: Oumeng Zhang*, Haowen Zhou*, Brandon Y. Feng, Elin M. Larsson, Reinaldo
  E. Alcalde, Siyuan Yin, Catherine Deng, Changhuei Yang

(In press at Advanced Photonics)

ArXiv paper version: https://arxiv.org/abs/2405.10463

Project Page: https://hwzhou2020.github.io/SVF-Web/

## Data source
Please download data and put them in the "data" folder
Link: https://osf.io/4a5ws/
'data' folder gives a few example data (all included in GitHub data folder).
'data_all' includes all data for the root sample.

##
fluo_recon_lymphn.py: simulation main code with lymph node vascular.
fluo_recon_root.py: experiments for plant root sample.

## Scripts
Example bash script to run the fluo_recon_root.py code

More scripts in the 'scripts' folder
```
CUDA_VISIBLE_DEVICES=0 python fluo_recon.py \
    --data_name Roots_xyScan_128.tif \
    --exp_psf_name ExpPSF_605.mat \
    --show_inter_imgs False \
    --model_opt "complie" \
    --if_log True \
    --wavelength 605e-9 \
    --lr_psf 5e-3 \
    --init_epochs 100 \
    --learn_psf_epochs 100 \
    --z_dim 8 \
    --z_sep 0.1 \
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
pytorch-msssim 
scipy
skimage
sys
torch
tqdm
```

## Reference / Citation
```
@misc{zhang2024SVF,
      title={Single-shot volumetric fluorescence imaging with neural fields}, 
      author={Oumeng Zhang and Haowen Zhou and Brandon Y. Feng and Elin M. Larsson and Reinaldo E. Alcalde and Siyuan Yin and Catherine Deng and Changhuei Yang},
      year={2024},
      eprint={2405.10463},
      archivePrefix={arXiv},
      primaryClass={physics.optics},
      url={https://arxiv.org/abs/2405.10463}, 
}
```






