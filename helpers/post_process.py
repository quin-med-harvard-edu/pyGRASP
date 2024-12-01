import numpy as np
from joblib import Parallel, delayed
from scipy.signal import resample
from tqdm import tqdm


def interpolate_slice(volume: np.ndarray,
                      img_size: tuple,
                      rate_os: float,
                      flag_fft_shift: bool = False,
                      flag_interpolate_z: bool = False):
    """
    Reconstructed image dimensions does not usually fit scanner reconstruction
    dimensions. This function replicates the operation in the scanner:
    Given a desired image size (img_size), the code crops x-y plane with img_size[0,1]
    and interpolates slice dimension to img_size[2] cropping oversampled z_dim.
    Args:
        volume(ndarray[complex64]): reconstructed volume of size NSl x Nv x NS x NS
            NSl, Nv, Ns = Number of slice x number of volume x number of sample
        img_size(tuple[int x int x int]): desired Ns, Ns, NSl
        rate_os(float): slice oversampling. If 1.2 number of slices will be input
            number x 1.2
        flag_fft_shift(bool): If true applies fft shift on slice level first
    return:
        volume(ndarray[complex64]): resampled volume of size NSl x Nv x NS x NS
    """

    [img_x, img_y, img_sl] = img_size

    if flag_fft_shift:
        volume = np.fft.fftshift(volume, axes=0)
    dim_vol = volume.shape

    # crop images to the center specified voxels
    if img_size is not None:
        x_change = int((dim_vol[2] - img_x) / 2)
        y_change = int((dim_vol[3] - img_y) / 2)

        if (x_change > 0) & (y_change > 0):
            yy, xx = np.meshgrid(
                np.arange(x_change, dim_vol[2] - x_change),
                np.arange(y_change, dim_vol[3] - y_change),
            )
            volume = volume[:, :, xx, yy]
        else:  # 0 padding
            padding = ((-x_change, -x_change), (-y_change, -y_change), (0, 0), (0, 0))
            volume = np.pad(volume, padding, "constant", constant_values=0)

    # oversampling
    if flag_interpolate_z:
        total_slices = np.round(img_sl * rate_os)

        # Interpolate the number of slices to match output dim
        vol_oversampled = np.array(Parallel(n_jobs=-2)(
            delayed(resample)(volume[:, idx_v, :, :], int(total_slices), axis=0) for
            idx_v in tqdm(range(volume.shape[1]))))
        # reshape to original form NSl x Nv x NS x NS
        vol_oversampled = np.transpose(vol_oversampled, (1, 0, 2, 3))

        # crop the samples that are outside of the range
        volume = vol_oversampled[range(int(np.floor(
            (total_slices - img_sl) / 2.)), int(img_sl + np.floor(
            (total_slices - img_sl) / 2.))), :, :, :]

    return volume
