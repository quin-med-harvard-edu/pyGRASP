import numpy as np


class NUFFTObject(object):
    def forward(
        self,
        batch_xk: np.ndarray,
        batch_k_samples: np.ndarray,
        batch_sqrt_dcf: np.ndarray,
        coil_p: np.ndarray,
    ):
        """Args:
        batch_xk(array[complex64]): [Sb x NS x NS] image domain repr.
        batch_k_samples(array[float]): [Sb x NS x NSp] star sampling
        batch_sqrt_dcf(array[complex]): [Sb x NS x NSp] square root of the dcf
        coil_p(array[complex]): [NS x NS x NC] coil profile matrix
            [Sb,NS,NC,NSp] = [size_batch, num_sample, num_coil, num_spoke]
        Return:
        out_(array[complex64]): [Sb x NS x NC x NSp] k_space data"""
        pass

    def adjoint(
        self,
        batch_k_space: np.ndarray,
        batch_k_samples: np.ndarray,
        batch_sqrt_dcf: np.ndarray,
        coil_p: np.ndarray,
    ):
        """Args:
        batch_k_space(array[complex64]): [Sb x NS x NC x NSp] k_space data
        batch_k_samples(array[float]): [Sb x NS x NSp] star sampling information
        batch_sqrt_dcf(array[complex]): [Sb x NS x NSp] square root of the dcf
        coil_p(array[complex]): [NS x NS x NC] coil profile matrix
            [Sb,NS,NC,NSp] = [size_batch, num_sample, num_coil, num_spoke]
        Return:
        out_(array[complex64]): [Sb x NC x NS x NS] image representation"""
        pass
