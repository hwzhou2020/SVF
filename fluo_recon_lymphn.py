import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import scipy.io as sio
from pytorch_msssim import SSIM

from network import DeconNet, FullModel
from util import extract_raw, Get_PSF, plotz, plot_deconvolution

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor') # torch version 


def get_args():
    parser = argparse.ArgumentParser()
    # Set general options
    parser.add_argument("--data_path", default='./data', type=str, help="path to data")
    parser.add_argument("--data_name", default='lymphn.tif', type=str, help=" simData.tif name of the raw image data")
    parser.add_argument("--show_inter_imgs", default=True, type=bool, help="show and save intermediate images")
    parser.add_argument("--display_freq", default=100, type=int, help="display / save intermediate image frequency, every n epochs")
    parser.add_argument("--out_dir", type=str, default="vis_exp")

    # Iterative deconvolution options
    parser.add_argument("--num_iters", default=100, type=int, help="number of iterations for deconvolution")
    parser.add_argument("--use_amp", default=True, type=bool, help="use automatic mixed precision for deconvolution")
    parser.add_argument("--model_opt", default="jit", type=str,help="options: [jit | compile | None]")

    # Learn PSF options
    parser.add_argument("--if_log", default=False, type=bool, help="logarithm of raw image")
    parser.add_argument("--if_lr_psf", default=True, help="if learn PSF")
    parser.add_argument("--learn_psf_epochs", default=100, type=int, help="number of epochs for learn psf")
    parser.add_argument("--init_epochs", default=100, type=int, help="number of epochs for initialization")
    parser.add_argument("--lr_psf", default=8e-3, type=float, help="learning rate for learn psf")
    parser.add_argument("--use_layernorm", default=False, type=bool, help="use layernorm in learn psf")

    # Data parameters 
    parser.add_argument("--patch_size", default=1024, type=int, help="image size")
    parser.add_argument("--z_max", default=2.5, type=float, help="maximum z-value in mm")
    parser.add_argument("--z_min", default=-2.5, type=float, help="minimum z-value in mm")
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


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_args()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    obj = sio.loadmat(os.path.join('data','lymphn.mat'))['obj']
    obj = obj - 1.332
    obj[obj<0] = 0
    obj = obj / np.max(obj)
    obj = resize(obj, (800, 800, 41), mode='reflect', anti_aliasing=True)
   
    # Target shape (1024, 1024, 41)
    target_shape = (1024, 1024, 41)
    current_shape = obj.shape

    # Symmetric padding for the first dimension
    pad_first_dim = (max(0, (target_shape[0] - current_shape[0]) // 2),
                    max(0, (target_shape[0] - current_shape[0]) - (target_shape[0] - current_shape[0]) // 2))

    # Left side padding for the second dimension
    pad_second_dim = (max(0, target_shape[1] - current_shape[1]), 0)

    # No padding for the third dimension
    pad_third_dim = (5, 5)

    # Combine the padding for all dimensions
    pad_width = [pad_first_dim, pad_second_dim, pad_third_dim]

    # Pad the object accordingly
    obj = np.pad(obj, pad_width=pad_width, mode='constant', constant_values=0)

    obj = torch.from_numpy(obj).float().to(device).unsqueeze(0)
    obj =  obj.permute(0, 3, 1, 2)
    g_true = obj.detach().cpu().numpy() * 1710.1 # (raw image scaling factor)

    plotz(g_true, out_dir, title="Stack GT")

    num_epochs = args.learn_psf_epochs + args.init_epochs
    imsize = int(args.patch_size + args.p_size)
    M = args.f_tube / args.f_obj  # magnification
    rBFP = args.NA * args.f_obj  # pupil radius
    rBFP_px = round(args.p_size * args.NA * args.px_size / args.wavelength / M)  # pupil diameter in pixels
    pad_pix = int(np.floor((args.p_size - 2*rBFP_px) / 2))
    pupil_size = int(rBFP_px * 2)

    # s and p stands for s and p polarizations
    PSF, PSFR, pupil_ampli_s, pupil_ampli_p, defocus = Get_PSF(M, rBFP, rBFP_px, args.px_size, args.wavelength, args.NA, args.block_line, args.pol_dir, args.z_min, args.z_max, args.z_sep, args.p_size)

    PSF = PSF.permute(2, 3, 0, 1)
    PSFR = PSFR.permute(2, 3, 0, 1) 
    PSF = PSF / PSF.sum() * args.num_pol * PSF.size(1)
    PSFR = PSFR / PSFR.sum() * args.num_pol * PSFR.size(1)

    num_z = PSF.shape[1]
    PSFsize = PSF.shape[2]

    imgTmp = io.imread(f'{args.data_path}/{args.data_name}').astype(np.float32)
    img = extract_raw(imgTmp, imsize, PSFsize)
    del imgTmp

    plt.figure(figsize=(10, 10), dpi=300)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(img[i, 0].cpu() / img.cpu().max(), cmap='gray')
        plt.axis('image')
        plt.clim(0, 1) 
        plt.axis('off')
        plt.title(f"p{i}")
    plt.suptitle('Raw images', fontsize=16, y=0.93)
    plt.savefig(f'{out_dir}/Raw_images.png', dpi=300)

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

    # show the deconvolved results
    plot_deconvolution(num_z, g, out_dir, tag='Theoretical PSF Deconvolution')
    del model, model_fn

    g_theo = g.clone().detach() 
    g = g_theo.clone().detach().to(torch.float32).to(device)
    #######################################################
    # Neural fields
    #######################################################

    psf_size, pad_size1 = PSF.shape[-1], g.shape[-1]  - PSF.shape[-1]

    if args.if_log:
        g_temp = torch.log(g_theo + 1)
        log_gmax, log_gmin = g_temp.max(), g_temp.min()
        g = (g_temp - log_gmin) / (log_gmax - log_gmin)
    
    dzs = torch.arange(args.z_min, args.z_max + args.z_sep, args.z_sep).to(torch.float32)
    dzs = dzs * 1e-3 # convert to m
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


    optimizer = torch.optim.AdamW(model_fn.parameters(), lr=args.lr_psf)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=args.lr_psf // 6)
    loss_fn = torch.nn.SmoothL1Loss()
    ssim_module = SSIM(data_range=1.0, size_average=False, channel=1)


    tbar = tqdm(range(num_epochs), desc='Optimization')
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
                plot_deconvolution(num_z, g, out_dir, tag='Initialized Deconvolution')
            optimizer.zero_grad()

            g_est = model_fn(dzs)
                                
            if args.if_log:
                g_est = g_est * (log_gmax - log_gmin) + log_gmin
                g_est = torch.exp(g_est) - 1

            F_g_est = torch.fft.fftn(g_est, dim=(-2, -1))

            pad_size1 = g.shape[-1]  - PSF.shape[-1]
            psf_padded = F.pad(PSF, (0, pad_size1,  0, pad_size1))
            psf_fft = torch.fft.fftn(psf_padded, dim=(-2, -1))

            FI_est = psf_fft * F_g_est.repeat(args.num_pol, 1, 1, 1)
            I_est_ext = torch.fft.ifftn(FI_est, dim=(-2, -1)).abs()

            # sum pooling
            I_est = torch.sum(I_est_ext, dim=1, keepdim=True)
            I_est = torch.roll(I_est, shifts=(-psf_size // 2 + 1, -psf_size // 2 + 1), dims=(2,3))

            im_loss = loss_fn(I_est, img) 
            ssim_loss = torch.sum((1 - ssim_module(I_est, img)) ** 2)

            lam = 2
            loss = im_loss + lam * ssim_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            # tbar.set_postfix(Loss = f'{im_loss.item():.3f}')


            if args.show_inter_imgs and (epoch + 1) % args.display_freq == 0:
                ext = 1
                sample_slice = int(num_z * ext)
                dz_sample = 1e-3 * torch.arange(args.z_min * ext, args.z_max * ext, ext * (args.z_max - args.z_min) / sample_slice).to(torch.float32)
                g_sample = model_fn(dz_sample)
                if args.if_log:
                    g_sample = g_sample * (log_gmax - log_gmin) + log_gmin
                    g_sample = torch.exp(g_sample) - 1
                
                plt.figure(figsize=(10, 10), dpi=300)
                for i in range(num_z):
                    plt.subplot(int(np.sqrt(num_z) + 1), int(np.sqrt(num_z) + 1), i + 1)
                    plt.imshow(np.clip(g_sample[0, i].detach().cpu().numpy(), 0, 1e3), cmap='gray')
                    plt.axis('image')
                    plt.axis('off')
                    plt.title(f"z{i}")
                plt.suptitle('Deconvolution', fontsize=16, y=0.17)
                plt.savefig(f'{out_dir}/Deconvolved_results_{epoch + 1}.png', dpi=300)
                plt.close()

                # Show estimated images
                plt.figure(figsize=(10, 10),dpi=300)
                for i in range(args.num_pol):
                    plt.subplot(2, 2, i+1)
                    plt.imshow(torch.relu(I_est[i, 0]).detach().cpu().numpy(), cmap='gray')
                    plt.axis('image')
                    plt.axis('off')
                    plt.title(f"p{i}")
                plt.suptitle('Estimated measurements', fontsize=16, y=0.93)
                plt.savefig(f'{out_dir}/Estimated_measurements_{epoch + 1}.png', dpi=300)
                plt.close()

    if args.if_log:
        g = g * (log_gmax - log_gmin) + log_gmin
        g = torch.exp(g) - 1

    plot_deconvolution(num_z, g_sample, out_dir, tag='Learned PSF Deconvolution')

    plotz(g.detach().cpu().numpy(), out_dir, title="Stack Deconvolution")

    plotz(g_sample.detach().cpu().numpy(), out_dir, title="Stack Nerf")
    

    sample_slice = int(num_z * 4)
    dz_sample = 1e-3 * torch.arange(args.z_min, args.z_max, (args.z_max - args.z_min) / sample_slice).to(torch.float32)
    g_interp = model_fn(dz_sample)
    if args.if_log:
        g_interp = g_interp * (log_gmax - log_gmin) + log_gmin
        g_interp = torch.exp(g_interp) - 1
                

    # sio.savemat(f'{out_dir}/LN_results.mat', {'g_deconv': g.detach().cpu().numpy(), 'g_nerf': g_sample.detach().cpu().numpy(), 'g_true': g_true, 'g_interp': g_interp.detach().cpu().numpy()})

    # save argparser to txt file
    with open(f'{out_dir}/args.txt', 'w') as f:
        for arg in vars(args):
            f.write(arg + ',' + str(getattr(args, arg)) + '\n')
