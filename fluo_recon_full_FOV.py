
if __name__ == "__main__":
    
    for fov in [127]:#[8, 11, 21, 41, 165, 178, 185]: #range(196):
        import os
        import sys
        import torch
        import argparse
        import numpy as np
        import torch.nn.functional as F
        from tqdm import tqdm
        from skimage import io
        import matplotlib.pyplot as plt
        import scipy.io as sio


        from network import DeconNet, FullModel
        from util import extract_raw, Get_PSF

        def get_args():
            parser = argparse.ArgumentParser()
            # Set general options
            parser.add_argument("--data_path", default='./data', type=str, help="path to data")
            parser.add_argument("--exp_psf_name", default='ExpPSF_605_20240425_shift.mat', type=str, help="experimental PSF name")
            parser.add_argument("--show_inter_imgs", default=False, type=bool, help="show and save intermediate images")
            parser.add_argument("--display_freq", default=100, type=int, help="display / save intermediate image frequency, every n epochs")
            parser.add_argument("--out_dir", type=str, default="vis_exp")
            parser.add_argument("--result_dir", type=str, default="results")

            # Iterative deconvolution options
            parser.add_argument("--num_iters", default=100, type=int, help="number of iterations for deconvolution")
            parser.add_argument("--use_amp", default=True, type=bool, help="use automatic mixed precision for deconvolution")
            parser.add_argument("--model_opt", default="jit", type=str,help="options: [jit | compile | None]")

            # Learn PSF options
            parser.add_argument("--if_log", default=False, type=bool, help="logarithm of raw image")
            parser.add_argument("--if_lr_psf", default=True, help="if learn PSF")
            parser.add_argument("--learn_psf_epochs", default=100, type=int, help="number of epochs for learn psf")
            parser.add_argument("--init_epochs", default=100, type=int, help="number of epochs for initialization")
            parser.add_argument("--lr_psf", default=2e-3, type=float, help="learning rate for learn psf")
            parser.add_argument("--use_layernorm", default=False, type=bool, help="use layernorm in learn psf")

            # Data parameters 
            parser.add_argument("--patch_size", default=1024, type=int, help="image size")
            parser.add_argument("--z_max", default=4.0, type=float, help="maximum z-value in mm")
            parser.add_argument("--z_min", default=-4.0, type=float, help="minimum z-value in mm")
            parser.add_argument("--z_sep", default=0.1, type=float, help="z separation in mm")
            parser.add_argument("--z_dim", default=8, type=int)
            parser.add_argument("--num_pol", default=4, type=int, help="number of polarizations")
            parser.add_argument("--p_size", default=256, type=int, help="pupil size with padding")

            # Learn PSF parameters
            parser.add_argument("--num_coeff", default=100, type=int, help="number of Zernike coefficients")
            parser.add_argument("--num_feats", default=32, type=int, help="number of features in the network")
            parser.add_argument("--l1_g", default=0.0, type=float, help="L1 sparsity weight for g_est")
            parser.add_argument("--l1_z", default=0.0, type=float, help="L1 sparsity weight for z_data")

            # Experimental parameters
            parser.add_argument("--wavelength", default=605e-9, type=float, help="emission wavelength")
            parser.add_argument("--px_size", default=6.9e-6, type=float, help="camera pixel size")
            parser.add_argument("--NA", default=.0563, type=float, help="numerical aperture")
            parser.add_argument("--f_obj", default=80e-3, type=float, help="objective lens focal length")
            parser.add_argument("--f_tube", default=150e-3, type=float, help="tube lens focal length")
            parser.add_argument("--block_line", default=.6e-3, type=float, help="3D printed pupil block center line width")
            parser.add_argument("--pol_dir", default=[1,2,3,0], type=int, help="polarizer direction, check this to match to the alignment of the polarizer")

            args = parser.parse_args()

            return args


        def abe_to_psf(aberration, num_pol, pupil_ampli_s, pupil_ampli_p, defocus):
            pupil_phase = (aberration + defocus).repeat(num_pol, 1, 1, 1)
            pupil_s = pupil_ampli_s * torch.exp(1j * pupil_phase) 
            pupil_p = pupil_ampli_p * torch.exp(1j * pupil_phase) 
            pupil_ifft = torch.flip(torch.fft.ifftshift(torch.fft.ifftn(pupil_s, dim=(-2, -1)), dim=(-2, -1)), dims=[-2, -1])
            psf_s = torch.abs(pupil_ifft) ** 2
            pupil_ifft = torch.flip(torch.fft.ifftshift(torch.fft.ifftn(pupil_p, dim=(-2, -1)), dim=(-2, -1)), dims=[-2, -1])
            psf_p = torch.abs(pupil_ifft) ** 2

            psf = psf_s + psf_p
            return psf


        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        torch.set_default_tensor_type('torch.cuda.FloatTensor') # torch version 

        torch.cuda.empty_cache()
        args = get_args()
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(args.result_dir, exist_ok=True)

        num_epochs = args.learn_psf_epochs + args.init_epochs
        imsize = int(args.patch_size + args.p_size)
        M = args.f_tube / args.f_obj  # magnification
        rBFP = args.NA * args.f_obj  # pupil radius
        rBFP_px = round(args.p_size * args.NA * args.px_size / args.wavelength / M)  # pupil diameter in pixels
        pad_pix = int(np.floor((args.p_size - 2*rBFP_px) / 2))
        pupil_size = int(rBFP_px * 2)

        # s and p stands for s and p polarizations
        PSF, PSFR, pupil_ampli_s, pupil_ampli_p, defocus = Get_PSF(M, rBFP, rBFP_px, args.px_size, args.wavelength, args.NA, args.block_line, args.pol_dir, args.z_min, args.z_max, args.z_sep, args.p_size)


        PSF = sio.loadmat(f'{args.data_path}/{args.exp_psf_name}')['PSF']
        PSFR = sio.loadmat(f'{args.data_path}/{args.exp_psf_name}')['PSFR']
        PSF = torch.tensor(PSF).to(torch.float32)
        PSFR = torch.tensor(PSFR).to(torch.float32)
        PSF = PSF.permute(2, 3, 0, 1)
        PSFR = PSFR.permute(2, 3, 0, 1) 
        PSF = PSF / PSF.sum() * args.num_pol * PSF.size(1)
        PSFR = PSFR / PSFR.sum() * args.num_pol * PSFR.size(1)

        num_z = PSF.shape[1]
        PSFsize = PSF.shape[2]

        print(str(fov+1)+ ' out of 196 FOVs')
        data_name = 'AVG_roots_xyzScan_z2_' + str(fov+1) + '.tif'
        imgTmp = io.imread(f'{args.data_path}/{data_name}').astype(np.float32)
        img = extract_raw(imgTmp, imsize, PSFsize)
        del imgTmp

        g = (img[0, 0] / num_z).repeat(num_z, 1, 1)[None].to(torch.float32)
        model = DeconNet(img, PSF, PSFR, PSFsize // 2, imsize, num_z).to(device)
        model_fn = model
        if args.model_opt == "jit":
            model_fn = torch.jit.trace(model, g)
        if args.model_opt == "compile":
            model_fn = torch.compile(model, backend="inductor")

        for iter in tqdm(range(args.num_iters)):
            if args.use_amp:
                with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=torch.float16):
                    g = model_fn(g)
            else:
                g = model_fn(g)
        g_exper = g
        # g_result_dir = f'{args.result_dir}/EXP_' + str(fov+1) + '.mat'
        # sio.savemat(g_result_dir, {"g_exp": g_exper.detach().cpu().numpy().astype('float16')}) ########################
        del g, model, model_fn, PSF, PSFR

        PSF, PSFR, pupil_ampli_s, pupil_ampli_p, defocus = Get_PSF(M, rBFP, rBFP_px, args.px_size, args.wavelength, args.NA, args.block_line, args.pol_dir, args.z_min, args.z_max, args.z_sep, args.p_size)
        dzs = 1e-3 * torch.arange(args.z_min, args.z_max + args.z_sep, args.z_sep).to(torch.float32)
        num_z = len(dzs)
        defocus_temp = defocus
        defocus = torch.tensor(defocus).to(torch.float32)[None, None].repeat(1, num_z, 1, 1)
        # multiply defocus phase term (1, num_z, im_size, imsize) by dzs (num_z, )
        defocus = defocus * dzs[..., None, None]
        defocus = defocus.to(device)

        PSF = PSF.permute(2, 3, 0, 1)
        PSFR = PSFR.permute(2, 3, 0, 1) # this is PSF rotated by 180 degrees
        num_z = PSF.shape[1]
        PSFsize = PSF.shape[2]

        # Deconvolution with retrieved phase PSF
        abe = sio.loadmat(f'{args.data_path}/{args.exp_psf_name}')['phase_init']
        pupil_ampli_s_temp = np.moveaxis(pupil_ampli_s, -1, 0)
        pupil_ampli_p_temp = np.moveaxis(pupil_ampli_p, -1, 0)
        # s-polarization
        pupil_ampli_s = np.moveaxis(pupil_ampli_s, -1, 0)
        pupil_ampli_s = torch.tensor(pupil_ampli_s).to(torch.float32)[:, None].repeat(1, num_z, 1, 1)
        pupil_ampli_s = pupil_ampli_s.to(device)
        # p-polarization
        pupil_ampli_p = np.moveaxis(pupil_ampli_p, -1, 0)
        pupil_ampli_p = torch.tensor(pupil_ampli_p).to(torch.float32)[:, None].repeat(1, num_z, 1, 1)
        pupil_ampli_p = pupil_ampli_p.to(device)
        
        #  to tensor
        abe = torch.tensor(abe).to(torch.float32).to(device) 
        PSF = abe_to_psf(abe, args.num_pol, pupil_ampli_s, pupil_ampli_p, defocus)
        PSF = PSF / PSF.sum() * args.num_pol * len(dzs)
        # rotate psf for 180 degree
        PSFR = (torch.flip(PSF, dims=[-2, -1]))

        g = (img[0, 0] / num_z).repeat(num_z, 1, 1)[None].to(torch.float32)
        model = DeconNet(img, PSF, PSFR, PSFsize // 2, imsize, num_z).to(device)
        model_fn = model
        if args.model_opt == "jit":
            model_fn = torch.jit.trace(model, g)
        if args.model_opt == "compile":
            model_fn = torch.compile(model, backend="inductor")

        for iter in tqdm(range(args.num_iters)):
            if args.use_amp:
                with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=torch.float16):
                    g = model_fn(g)
            else:
                g = model_fn(g)
        g_retri = g
        g_result_dir = f'{args.result_dir}/RET_' + str(fov+1) + '.mat'
        sio.savemat(g_result_dir, {"g_ret":g_retri.detach().cpu().numpy().astype('float16')})
        del g, model, model_fn, PSF, PSFR, pupil_ampli_s, pupil_ampli_p, defocus

        
        #######################################################
        # Neural representations
        dzs_ret = torch.arange(args.z_min, args.z_max + args.z_sep, args.z_sep).to(torch.float32)
        dzs_exp = torch.arange(-2.0, 2.0 + 0.1, 0.1).to(torch.float32)

        idx_eInr = [torch.argmin(torch.abs(dzs_ret - dz)).item() for dz in dzs_exp if torch.min(torch.abs(dzs_ret - dz)) < 0.001]
        idx_eIne = [torch.argmin(torch.abs(dzs_exp - dz)).item() for dz in dzs_ret if torch.min(torch.abs(dzs_exp - dz)) < 0.001]

        g_retri[0,idx_eInr] = g_exper[0,idx_eIne]
        idx_eNotInr = [i for i in range(len(dzs_exp)) if i not in idx_eIne]

        dzs = 1e-3 * torch.cat((dzs_exp[idx_eNotInr], dzs_ret))
        g = torch.cat((g_exper[0,idx_eNotInr], g_retri[0]), dim=0).unsqueeze(0)
        dzs, idx = torch.sort(dzs)
        g = g[:,idx]          

        if args.if_log:
            g = torch.log(g + 1)
            log_gmax, log_gmin = g.max(), g.min()
            g = (g - log_gmin) / (log_gmax - log_gmin)
        
        num_z = len(dzs)
        dzs = dzs.to(device)

        model = FullModel(
            w = g.shape[-1], 
            h = g.shape[-1], 
            num_feats = args.num_feats, 
            x_mode = g.shape[-1], 
            y_mode = g.shape[-2],
            z_min = dzs[0], 
            z_max = dzs[-1], 
            z_dim = args.z_dim,
            ds_factor = 2, 
            use_layernorm = args.use_layernorm
        ).to(device)

        model_fn = model
        if args.model_opt == "jit":
            model_fn = torch.jit.trace(model, dzs)
        if args.model_opt == "compile":
            model_fn = torch.compile(model, backend="inductor")

        # Set Pupil amplitude
        pupil_ampli_s = torch.tensor(pupil_ampli_s_temp).to(torch.float32)[:, None].repeat(1, num_z, 1, 1).to(device)
        pupil_ampli_p = torch.tensor(pupil_ampli_p_temp).to(torch.float32)[:, None].repeat(1, num_z, 1, 1).to(device)

        abe = sio.loadmat(f'{args.data_path}/{args.exp_psf_name}')['phase_init']
        abe = torch.tensor(abe).to(torch.float32).to(device) 

        optimizer = torch.optim.AdamW(model_fn.parameters(), lr=args.lr_psf)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=args.lr_psf // 6)
        loss_fn = torch.nn.SmoothL1Loss()

        tbar = tqdm(range(num_epochs), desc='Learn PSF')
        for epoch in tbar:

            if epoch < args.init_epochs:
            
                optimizer.zero_grad()
                g_est = model_fn(dzs) # sample the model with predefined dzs
                
                im_loss = loss_fn(g_est, g) 
                loss = im_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                tbar.set_postfix(Loss = f'{im_loss.item():.3f}')

            else:
                if epoch == args.init_epochs:
                    PSF = sio.loadmat(f'{args.data_path}/{args.exp_psf_name}')['PSF']
                    PSFR = sio.loadmat(f'{args.data_path}/{args.exp_psf_name}')['PSFR']
                    PSF = torch.tensor(PSF).to(torch.float32)
                    PSFR = torch.tensor(PSFR).to(torch.float32)
                    PSF = PSF.permute(2, 3, 0, 1)
                    PSFR = PSFR.permute(2, 3, 0, 1) 

                    exp_idx = [torch.argmin(torch.abs(dzs - dz)).item() for dz in dzs if torch.min(torch.abs(dzs_exp*1e-3 - dz)) < 1e-5]
                    ret_idx = [i for i in range(len(dzs)) if i not in exp_idx]
                    
                optimizer.zero_grad()
                dzs_sample = ( 1e-3 * dzs[exp_idx] + 0.5e-4 * torch.rand(len(dzs[exp_idx])) ).to(torch.float32)
                dzs_ext = (1e-3 * dzs[ret_idx] + 0.5e-4 * torch.rand(len(dzs[ret_idx])) ).to(torch.float32)

                g_est = model_fn(dzs)
                g_ret_sample = g_est[:, ret_idx]
                g_exp_sample = g_est[:, exp_idx]

                z_data = model.model_3D.img_real.z_data
                sparsity_loss = args.l1_z * torch.mean(z_data.abs()) + args.l1_g * torch.mean(g_est.abs())

                g_est = g_exp_sample
                if args.if_log:
                    g_est = g_est * (log_gmax - log_gmin) + log_gmin
                    g_est = torch.exp(g_est) - 1
                F_g_est = torch.fft.fftn(g_est, dim=(-2, -1))

                # experimental PSF
                psf_size, pad_size1 = PSF.shape[-1], g.shape[-1]  - PSF.shape[-1]
                psf_padded = F.pad(PSF, (0, pad_size1,  0, pad_size1))
                psf_fft = torch.fft.fftn(psf_padded, dim=(-2, -1))

                FI_est = psf_fft * F_g_est.repeat(args.num_pol, 1, 1, 1)
                I_est_sample = torch.fft.ifftn(FI_est, dim=(-2, -1)).abs()

                # retrieved PSF
                g_est = g_ret_sample
                
                if args.if_log:
                    g_est = g_est * (log_gmax - log_gmin) + log_gmin
                    g_est = torch.exp(g_est) - 1
                F_g_est = torch.fft.fftn(g_est, dim=(-2, -1))


                # Set defocus phase term
                defocus_ext = torch.tensor(defocus_temp).to(torch.float32)[None, None].repeat(1, len(dzs_ext), 1, 1)
                # multiply defocus phase term (1, num_z, im_size, imsize) by dzs (num_z, )
                defocus_ext = defocus_ext * dzs_ext[..., None, None]
                defocus_ext = defocus_ext.to(device)
                
                pupil_ampli_s = pupil_ampli_s[:, :len(dzs_ext)] 
                pupil_ampli_p = pupil_ampli_p[:, :len(dzs_ext)] 

                psf = abe_to_psf(abe, args.num_pol, pupil_ampli_s, pupil_ampli_p, defocus_ext)
                psf = psf / psf.sum() * args.num_pol * psf.size(1)
                psf_size, pad_size1 = psf.shape[-1], g.shape[-1]  - psf.shape[-1]
                psf_padded = F.pad(psf, (0, pad_size1,  0, pad_size1))
                psf_fft = torch.fft.fftn(psf_padded, dim=(-2, -1))

                FI_est = psf_fft * F_g_est.repeat(args.num_pol, 1, 1, 1)
                I_est_ext = torch.fft.ifftn(FI_est, dim=(-2, -1)).abs()

                # concatenate the two results
                I_est = torch.cat((I_est_sample, I_est_ext), dim=1)
                # sum pooling
                I_est = torch.sum(I_est, dim=1, keepdim=True)
                I_est = torch.roll(I_est, shifts=(-psf_size // 2 + 1, -psf_size // 2 + 1), dims=(2,3))

                im_loss = loss_fn(I_est, img) / img.mean()
                loss = im_loss + sparsity_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                tbar.set_postfix(Loss = f'{im_loss.item():.3f}')

        if args.if_log:
            g = g * (log_gmax - log_gmin) + log_gmin
            g = torch.exp(g) - 1
        

        ext = 1
        sample_slice = int(num_z * ext)
        dz_sample = 1e-3 * torch.arange(args.z_min * ext, args.z_max * ext, ext * (args.z_max - args.z_min) / sample_slice).to(torch.float32)
        g_sample = model_fn(dz_sample)
        if args.if_log:
            g_sample = g_sample * (log_gmax - log_gmin) + log_gmin
            g_sample = torch.exp(g_sample) - 1

        g_result_dir = f'{args.result_dir}/INR_' + str(fov+1) + '.mat'
        sio.savemat(g_result_dir, {"g_inr": g_sample.detach().cpu().numpy().astype('float16')})
        del g, model, model_fn, PSF, PSFR, pupil_ampli_s, pupil_ampli_p, psf
        del g_sample, I_est, I_est_sample, I_est_ext, dzs, dzs_ret, dzs_exp, dzs_ext, dzs_sample
        del g_retri, g_exper, img, scheduler, optimizer, loss_fn

        # save argparser to txt file
        with open(f'{out_dir}/args.txt', 'w') as f:
            for arg in vars(args):
                f.write(arg + ',' + str(getattr(args, arg)) + '\n')

        # Get the names of imported modules/packages
        imported_packages = sys.modules.keys()
        # Delete all variables except imported packages
        for var in list(locals().keys()):
            if var not in imported_packages:
                del locals()[var]

