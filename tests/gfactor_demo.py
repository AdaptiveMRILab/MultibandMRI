import argparse
import os
import h5py
import numpy as np
import torch
from MultibandMRI import slice_grappa, sense_grappa, fft2d, ifft2d, CoilCompress
import matplotlib.pyplot as plt

def load_sms_dataset(sms_folder="data", sms_factor=3, device="cpu", ncoils=None, crop=None):
    calibfile = os.path.join(sms_folder, f"SMS_{sms_factor}", "ksp_calib.h5")
    accelfile = os.path.join(sms_folder, f"SMS_{sms_factor}", "ksp_accel.h5")

    with h5py.File(calibfile, "r") as f:
        ksp_calib = torch.tensor(np.array(f["ksp"][..., 0], dtype=np.complex64), dtype=torch.complex64, device=device)

    N = ksp_calib.shape[0]

    if sms_factor % 2 == 0:
        phi = 2 * np.pi / (2 * sms_factor)
        for p in range(2 * sms_factor):
            ksp_calib[:, p::(2 * sms_factor), :] = ksp_calib[:, p::(2 * sms_factor), :] * np.exp(-1j * p * phi)

    img = ifft2d(ksp_calib, dims=(0, 1))
    isocenter_slice = sms_factor // 2
    nshift = (sms_factor - isocenter_slice - 1) * N
    img = torch.roll(img, shifts=nshift, dims=1)
    img = torch.stack([img[:, (n * N):((n + 1) * N), :] for n in range(sms_factor)], axis=-1)
    img = torch.flip(img, dims=(-1,))
    data = fft2d(img, dims=(0, 1))

    blip_phase_increment = 2 * np.pi / sms_factor
    for slc in range(sms_factor):
        for p in range(sms_factor):
            data[:, p::sms_factor, :, slc] = data[:, p::sms_factor, :, slc] * np.exp(1j * blip_phase_increment * slc * p)

    calib_data = data.permute(3, 2, 0, 1).contiguous()  # (sms, coils, ro, pe)

    with h5py.File(accelfile, "r") as f:
        aliased_ksp = np.array(f["ksp"], dtype=np.complex64)[:, :, :, 0]
    aliased_ksp = torch.tensor(aliased_ksp, dtype=torch.complex64, device=device)
    aliased_ksp = aliased_ksp.permute(2, 0, 1)[None, ...].contiguous()

    # optional coil compression (shared basis from calibration)
    if ncoils is not None and ncoils < calib_data.shape[1]:
        # CoilCompress expects shape (batch, coils, ro, pe)
        compressor = CoilCompress(calib_data, ncoils)
        calib_data = compressor.compress(calib_data)
        aliased_ksp = compressor.compress(aliased_ksp)

    # optional spatial crop (center crop on read/phase)
    if crop is not None:
        ro_crop, pe_crop = crop
        def center_crop(x):
            ro, pe = x.shape[-2], x.shape[-1]
            r0 = max((ro - ro_crop) // 2, 0)
            p0 = max((pe - pe_crop) // 2, 0)
            return x[..., r0:r0+ro_crop, p0:p0+pe_crop]
        calib_data = center_crop(calib_data)
        aliased_ksp = center_crop(aliased_ksp)

    # match coil dimension for accel data
    if ncoils is not None and ncoils < aliased_ksp.shape[1]:
        aliased_ksp = aliased_ksp[:, :ncoils, ...]

    accel_phase = calib_data.shape[-1] // aliased_ksp.shape[-1]
    accel = (1, accel_phase)
    return calib_data, aliased_ksp, accel


def main():
    parser = argparse.ArgumentParser(description="G-factor demo on provided SMS dataset.")
    parser.add_argument("--sms_folder", default="data")
    parser.add_argument("--sms_factor", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kernel_r", type=int, default=5)
    parser.add_argument("--kernel_p", type=int, default=5)
    parser.add_argument("--ncoils", type=int, default=8, help="Use first N coils (truncate).")
    parser.add_argument("--crop", type=int, nargs=2, default=[128, 128], help="Center crop ro pe for faster demo.")
    args = parser.parse_args()

    calib, accel_ksp, accel = load_sms_dataset(args.sms_folder, args.sms_factor, args.device, ncoils=args.ncoils, crop=tuple(args.crop) if args.crop else None)
    print(f"Loaded calib {tuple(calib.shape)}, accel {tuple(accel_ksp.shape)}, accel factors {accel}")

    obj = slice_grappa(calib, accel=accel, kernel_size=(args.kernel_r, args.kernel_p), final_matrix_size=calib.shape[2:])
    ksp_recon, rss, gmap = obj.apply(accel_ksp, return_gfactor=True)

    plt.figure()
    plt.imshow(rss[0,:,:].cpu(), cmap='gray')
    plt.show()

    print(rss.shape)

    plt.figure()
    plt.imshow(gmap[0,:,:].cpu(), vmin=0.0, vmax=2.0)
    plt.show()

    print("Recon kspace shape:", tuple(ksp_recon.shape))
    print("RSS shape:", tuple(rss.shape))
    print("G-factor stats -> mean: {:.3f}, min: {:.3f}, max: {:.3f}".format(
        gmap.mean().item(), gmap.min().item(), gmap.max().item()))


if __name__ == "__main__":
    main()
