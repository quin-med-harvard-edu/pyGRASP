import numpy as np


def down_sample_freq(
    k_space: np.ndarray,
    k_samples: np.ndarray,
    sqrt_dcf: np.ndarray,
    coil_p: np.ndarray,
    factor: int = None,
):
    """This function downsamples the frequency components on the input and crops
    image domain data with respect to a predefined factor
    Args:
        k_space(ndarray[complex]): [Nv x NS x NC x NSp] k_space observation
        k_samples(ndarray[float]): [Nv x NS x NSp] star sampling information
        sqrt_dcf(ndarray[complex]): [Nv x NS x NSp] square root of the dcf matrix
        coil_p(ndarray[complex]): [NS x NS x NC] coil profile matrix
            [Nv,NS,NC,NSp] = [num_volume, num_sample, num_coil, num_spoke]
        factor(int): downsampling in frequency domain factor (fov scaling on image)
    Return: Rescaled versions of
        k_space(ndarray[complex]): [Nv x NS x NC x NSp] k_space observation
        k_samples(ndarray[float]): [Nv x NS x NSp] star sampling information
        sqrt_dcf(ndarray[complex]): [Nv x NS x NSp] square root of the dcf matrix
        coil_p(ndarray[complex]): [NS x NS x NC] coil profile matrix"""

    if factor:
        k_space = k_space[:, ::factor, :, :]
        k_samples = k_samples[:, ::factor, :]
        sqrt_dcf = sqrt_dcf[:, ::factor, :]
        side_crop = int(coil_p.shape[0] * (1 - (1 / factor)) / 2)
        coil_p = coil_p[
            side_crop : coil_p.shape[0] - side_crop,
            side_crop : coil_p.shape[1] - side_crop,
            :,
        ]

    return k_space, k_samples, sqrt_dcf, coil_p


def divide_in_time(
    k_space: np.ndarray,
    k_samples: np.ndarray,
    sqrt_dcf: np.ndarray,
    num_max_spoke: int = None,
    spv: int = 34,
):
    """This function downsamples the frequency components on the input and crops
    image domain data with respect to a predefined factor
    Args:
        k_space(ndarray[complex]): [NS x NC x NSp] k_space observation
        k_samples(ndarray[float]): [NS x NSp] star sampling information
        sqrt_dcf(ndarray[complex]): [NS x NSp] square root of the dcf matrix
        num_max_spoke(int): limits number of volumes to be generated
        spv(int): number of spokes for one time instance reconstruction
    Return: Rescaled versions of
        div_k_space(ndarray[complex]): [Nv x NS x NC x NSp] k_space observation
        div_k_samples(ndarray[float]): [Nv x NS x NSp] star sampling information
        div_sqrt_dcf(ndarray[complex]): [Nv x NS x NSp] square root of the dcf matrix
            [Nv,NS,NC,NSp] = [num_volume, num_sample, num_coil, num_spoke]"""

    if num_max_spoke == None:
        num_max_spoke = k_space.shape[2]

    num_vol = int(np.floor(num_max_spoke / spv))

    div_k_space = np.array(
        [k_space[:, :, el * spv : (el + 1) * spv] for el in range(num_vol)]
    )
    div_k_samples = np.array(
        [k_samples[:, el * spv : (el + 1) * spv] for el in range(num_vol)]
    )
    div_sqrt_dcf = np.array(
        [sqrt_dcf[:, el * spv : (el + 1) * spv] for el in range(num_vol)]
    )

    return div_k_space, div_k_samples, div_sqrt_dcf
