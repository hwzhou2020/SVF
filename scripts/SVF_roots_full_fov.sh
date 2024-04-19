CUDA_VISIBLE_DEVICES=0 python fluo_recon_full_FOV.py \
    --exp_psf_name ExpPSF_605_20240311_shift.mat \
    --show_inter_imgs False \
    --model_opt "complie" \
    --if_log True \
    --wavelength 605e-9 \
    --lr_psf 2e-3 \
    --init_epochs 100 \
    --learn_psf_epochs 100 \
    --z_dim 8 \
    --z_sep 0.1 \
    --l1_g 0 \
    --l1_z 0

    # --l1_g 1e3 \
    # --l1_z 5e2 \