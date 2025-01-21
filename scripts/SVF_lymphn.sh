CUDA_VISIBLE_DEVICES=1 python fluo_recon_lymphn.py \
    --data_name lymphn.tif \
    --model_opt "complie" \
    --if_log True \
    --wavelength 605e-9 \
    --lr_psf 5e-3 \
    --num_iters 100 \
    --init_epochs 100 \
    --learn_psf_epochs 100 \
    --z_dim 8 \
    --z_sep 0.1 
    
