from copy import copy

import numpy as np
from helpers.nufft.base_nufft import NUFFTObject
from joblib import Parallel, delayed
from pynufft import NUFFT


class NpNUFFT(NUFFTObject):
    def __init__(
        self,
        batch_k_samples: np.ndarray,
        nd=(448, 448),
        kd=(int(448 * 1.5), int(448 * 1.5)),
        jd=(2, 2),
    ):
        self.list_planned_nufft = Parallel(n_jobs=-2)(
            delayed(plan_np_nufft)(
                np.squeeze(batch_k_samples[i, :, :]), nd=nd, kd=kd, jd=jd
            )
            for i in range(len(batch_k_samples))
        )

    def forward(
        self,
        batch_xk: np.ndarray,
        batch_k_samples: np.ndarray,
        batch_sqrt_dcf: np.ndarray,
        coil_p: np.ndarray,
    ):
        k = np.array(
            Parallel(n_jobs=-2)(
                delayed(np_nufft)(
                    batch_xk[i, :, :],
                    batch_k_samples[i, :, :],
                    batch_sqrt_dcf[i, :, :],
                    coil_p,
                    self.list_planned_nufft[i],
                )
                for i in range(batch_k_samples.shape[0])
            )
        )
        return k

    def adjoint(
        self,
        batch_k_space: np.ndarray,
        batch_k_samples: np.ndarray,
        batch_sqrt_dcf: np.ndarray,
        coil_p: np.ndarray,
    ):
        xk = np.array(
            Parallel(n_jobs=-2)(
                delayed(np_inverse_nufft)(
                    batch_k_space[i, :, :],
                    batch_k_samples[i, :, :],
                    batch_sqrt_dcf[i, :, :],
                    coil_p,
                    copy(self.list_planned_nufft[i]),
                )
                for i in range(len(batch_k_samples))
            )
        )
        return xk


def plan_np_nufft(
    k_samples: np.ndarray, nd=(448, 448), kd=(int(448 * 1.5), int(448 * 1.5)), jd=(2, 2)
):
    """Plan Non-uniform Fourier transform using the k-samples information
    Args:
        k_samples(ndarray[float]): [NS x NSp] star sampling information
        pynuffy lib-specific parameters
            nd(tuple(int,int)):
            kd(tuple(int,int)):
            jd(tuple(int,int)):
    Return:
        nufft_obj: planned pynuffy object
    """
    nufft_obj = NUFFT()
    om = k_samples[:, :]
    om = [np.real(om.ravel()) * 2 * np.pi, np.imag(om.ravel()) * 2 * np.pi]
    nufft_obj.plan(np.array(om).T, nd, kd, jd)
    return nufft_obj


def np_inverse_nufft(
    k_space: np.ndarray,
    k_samples: np.ndarray,
    sqrt_dcf: np.ndarray,
    coil_p: np.ndarray,
    planned_nufft: object = None,
    nd=(448, 448),
    kd=(int(448 * 1.5), int(448 * 1.5)),
    jd=(6, 6),
):
    """Single slice NUFFT operator utilizing numpy and NUFFT library
    Do not use for deep learning purposes!
    Args:
        k_space(ndarray[complex]): [NS x NC x NSp] k_space data
        k_samples(ndarray[float]): [NS x NSp] star sampling information
        sqrt_dcf(ndarray[complext]): [NS x NSp] square root of the dcf matrix
        coil_p(ndarray[complex]): [NS x NS x NC] coil profile matrix
            [NS,NC,NSp] = [num_sample, num_coil, num_spoke]
        planned_nufft(object(NUFFT)): pre planned NUFFT object
        pynuffy lib-specific parameters
            nd(tuple(int,int)):
            kd(tuple(int,int)):
            jd(tuple(int,int)):
    Return:
        xk(ndarray[complex]): [NC x NS x NS] image domain representation
    """
    [im_size_1, im_size_2, num_coil] = coil_p.shape
    [_, num_sample] = k_samples.shape

    if planned_nufft:
        nufft_obj = planned_nufft
    else:
        nufft_obj = plan_np_nufft(k_samples, nd, kd, jd)

    xk = np.zeros((num_coil, im_size_1, im_size_2), dtype=np.complex64)
    for idx_coil in range(num_coil):  # for all channels
        tmp_ = nufft_obj.adjoint(
            (k_space[:, idx_coil, :] * sqrt_dcf).ravel()
        ) * np.conj(coil_p[:, :, idx_coil])
        # pynufft follows the np.fft conventions normalizing with data size.
        # however, for the implementation we need normalize with sqrt of the data sizes.
        # therefore multiply with the sqrt of the entire data size and with pi/2
        tmp_ *= (np.pi * np.sqrt(im_size_1 * im_size_2 * 2)) / 2
        # TODO:Understand why
        # Compensation
        tmp_ *= (im_size_1 * np.pi) / (2 * num_sample)

        xk[idx_coil, :, :] = copy(tmp_)

    xk /= np.sum(np.abs(coil_p) ** 2, axis=2)

    return np.array(xk, dtype=np.complex64)  # NS x NS


def np_nufft(
    xk: np.ndarray,
    k_samples: np.ndarray,
    sqrt_dcf: np.ndarray,
    coil_p: np.ndarray,
    planned_nufft: object = None,
    nd=(448, 448),
    kd=(int(448 * 1.5), int(448 * 1.5)),
    jd=(2, 2),
):
    """Single slice inverse NUFFT operator utilizing numpy and NUFFT library
    Do not use for deep learning purposes!
    Args:
        xk(ndarray[complex]): [NS x NS] image domain representation
        k_samples(ndarray[float]): [NS x NSp] star sampling information
        sqrt_dcf(ndarray[complext]): [NS x NSp] square root of the dcf matrix
        coil_p(ndarray[complex]): [NS x NS x NC] coil profile matrix
            [NS,NC,NSp] = [num_sample, num_coil, num_spoke]
        planned_nufft(object(NUFFT)): pre planned NUFFT object
        pynuffy lib-specific parameters
            nd(tuple(int,int)):
            kd(tuple(int,int)):
            jd(tuple(int,int)):
    Return:
        k_space(ndarray[complex]): [NS x NC x NSp] k_space data
    """
    [num_sample, num_spoke] = sqrt_dcf.shape
    [im_size_1, im_size_2, num_coil] = coil_p.shape

    if planned_nufft:
        nufft_obj = planned_nufft
    else:
        nufft_obj = plan_np_nufft(k_samples, nd, kd, jd)

    k_space = list(range(num_coil))
    for idx_coil in range(num_coil):  # for all channels
        k_space[idx_coil] = np.complex64(
            nufft_obj.forward(coil_p[:, :, idx_coil] * xk[:, :]).reshape(
                num_sample, num_spoke
            )
            * sqrt_dcf
        ) / np.sqrt(im_size_1 * im_size_2)

    return np.array(k_space).transpose((1, 0, 2))  # NS x NC x NSp
