# Function: utility functions

import os
import torch
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter

def extract_raw(imgTmp, imsize, PSFsize):
    # extract polarized images from raw image
    img = torch.zeros(imgTmp.shape[0]//2, imgTmp.shape[1]//2, 4)

    img[:, :, 0] = torch.from_numpy(imgTmp[::2, ::2]) # 0deg
    img[:, :, 1] = torch.from_numpy(imgTmp[::2, 1::2]) # 45deg
    img[:, :, 2] = torch.from_numpy(imgTmp[1::2, 1::2]) # 90deg
    img[:, :, 3] = torch.from_numpy(imgTmp[1::2, ::2] ) # 135deg

    img = img[:imsize-PSFsize,:imsize-PSFsize].to(torch.float32)
    img = img[:imsize-PSFsize,:imsize-PSFsize].permute(2,0,1)
    img = img.unsqueeze(1)

    imgTmp = img.cpu().numpy()
    # Noise reduction
    imgTmp = median_filter(imgTmp, size=(1, 1, 5, 5))
    img = torch.tensor(imgTmp).to(torch.float32)

    return img

def rgb2hls(rgb):
    return np.array([colorsys.rgb_to_hls(*color[:3]) for color in rgb])

def hls2rgb(hls):
    return np.array([colorsys.hls_to_rgb(*color) for color in hls])

def cMapHeight_hsv_v2(N, s):
    hsv_map = plt.cm.hsv(np.linspace(0, 1, 900))[:600, :3]
    hls_map = rgb2hls(hsv_map)
    hls_map[:, 1] = s # Modify the lightness in HLS
    rgb_map = hls2rgb(hls_map)
    idx = np.round(np.linspace(0, 599, N)).astype(int)
    out_cmap = rgb_map[idx]
    return out_cmap

def plotz(g, title="Stack Ground Truth"):
    cMap = cMapHeight_hsv_v2(g.shape[1], .6)
    gPlot = np.zeros((g.shape[2],g.shape[3],3))

    for i in range(g.shape[1]):
        colorized_channel = np.stack([
            cMap[i, 0] * g[0, i, :, :],
            cMap[i, 1] * g[0, i, :, :],
            cMap[i, 2] * g[0, i, :, :]
        ], axis=-1)
        gPlot += colorized_channel

    # added gaussian blur otherwise they are hard to see...
    if title == "Stack Ground Truth":
        gPlot_filt = gaussian_filter(gPlot, sigma=5, axes=(0,1))
        gPlot_filt = gPlot_filt/np.max(gPlot_filt)
    else:
        gPlot_filt = gPlot/np.max(gPlot)

    return gPlot_filt

def plot_deconvolution(num_z, g, out_dir,tag='Theoretical PSF Deconvolution'):
    plt.figure(figsize=(10, 10), dpi=300)
    for i in range(num_z):
        plt.subplot(int(np.sqrt(num_z) + 1), int(np.sqrt(num_z) + 1), i + 1)
        plt.imshow(np.clip(g[0, i].detach().cpu().numpy(), 0, 1e3), cmap='gray')
        plt.axis('image')
        plt.axis('off')
        plt.title(f"z{i}")
    plt.suptitle('Deconvolution', fontsize=16, y=0.17)
    plt.savefig(f'{out_dir}/Results '+tag+'.png', dpi=300)

    return None

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi,rho)

def Get_PSF(M, rBFP, rBFP_px, px_size, wavelength, NA, block_line, pol_dir, z_min, z_max, z_sep, p_size=256, num_pol=4):

    block_line_px = round(rBFP_px / rBFP / 2 * block_line)  # 3center line half width in pixels

    u, v = np.meshgrid(np.linspace(-wavelength / (2 * px_size / M), wavelength / (2 * px_size / M), p_size),
                    np.linspace(-wavelength / (2 * px_size / M), wavelength / (2 * px_size / M), p_size))
    p, r = np.arctan2(v, u), np.hypot(u, v)

    # compute amplitude mask A
    Atmp = np.ones((p_size, p_size))
    Atmp[r >= NA] = 0
    Atmp[p_size // 2 - block_line_px:p_size // 2 + block_line_px, :] = 0
    Atmp[:, p_size // 2 - block_line_px:p_size // 2 + block_line_px] = 0

    pupil_amplitude_s_pol = np.zeros((p_size, p_size, num_pol))
    pupil_amplitude_p_pol = np.zeros((p_size, p_size, num_pol))
    for i in range(num_pol):
        pupil_amplitude_s_pol[:, :, i] = Atmp * np.rot90(np.block([[np.ones((p_size // 2, p_size // 2)), 0.5 * np.ones((p_size // 2, p_size // 2))],
                                                                [0.5 * np.ones((p_size // 2, p_size // 2)), np.zeros((p_size // 2, p_size // 2))]]),
                                                                k=pol_dir[i])
        pupil_amplitude_p_pol[:, :, i] = Atmp * np.rot90(np.block([[np.zeros((p_size // 2, p_size // 2)), 0.5 * np.ones((p_size // 2, p_size // 2))],
                                                                [0.5 * np.ones((p_size // 2, p_size // 2)), np.zeros((p_size // 2, p_size // 2))]]),
                                                                k=pol_dir[i])
        
    # defocus phase
    p_defocus = 2 * np.pi / wavelength * np.cos(np.arcsin(r))
    p_defocus = p_defocus - np.min(p_defocus)

    # FT to find PSF
    # for z in ... A[:,:,0] -- first channel, etc.
    dzs = 1e-3 * np.arange(z_min, z_max + z_sep, z_sep)
    # print(len(dzs))
    PSF = np.zeros((p_size, p_size, num_pol, len(dzs)))
    
    for pol in range(num_pol):
        count = 0
        for z in dzs:
            PSF[:,:,pol,count] = np.abs(np.fft.fftshift(np.fft.fft2(pupil_amplitude_s_pol[:, :, pol] * np.exp(1j * p_defocus * z))))**2 + np.abs(np.fft.fftshift(np.fft.fft2(pupil_amplitude_p_pol[:, :, pol] * np.exp(1j * p_defocus * z))))**2
            count += 1

    PSF = PSF / np.sum(PSF) * num_pol * len(dzs)

    # rotate PSF for 180 deg at each plane (axis=2) [N N num_pol 31]
    PSFR = np.rot90(PSF, k=2, axes=(0,1))

    return torch.tensor(PSF.copy()).to(torch.float32), torch.tensor(PSFR.copy()).to(torch.float32), pupil_amplitude_s_pol, pupil_amplitude_p_pol, p_defocus
