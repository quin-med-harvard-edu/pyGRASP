from copy import copy

import numpy as np
import torch
import torchkbnufft as tkbn
from helpers.nufft.base_nufft import NUFFTObject


class PTNufft(NUFFTObject):
    """A Non uniform Fast Fourier Transform object
    This object only wraps the library: https://github.com/mmuckley/torchkbnufft
    Attr:
        operator: tkbn.KbNufft forward NUFFT operator
        adj_operator:tkbn.KbNufftAdjoint adjoint NUFFT operator
    Methods:
        adjoint_op: adjoint NUFFT operator that wraps all pre-requisites for MRI
        forward_op: forward NUFFT operator that wraps all pre-requisites for MRI
    """

    def __init__(
        self,
        nd=(448, 448),
        kd=(int(448 * 1.5), int(448 * 1.5)),
        jd=(2, 2),
        to_cuda: bool = True,
        size_batch: int = 10,
    ):
        """Args:
        to_cuda(bool): with access to gpu, if set true map all to GPU
        size_batch(int): number of volumes to pass to GPU in a single step"""
        self.operator = tkbn.KbNufft(im_size=nd, grid_size=kd, numpoints=jd)
        self.adj_operator = tkbn.KbNufftAdjoint(im_size=nd, grid_size=kd, numpoints=jd)

        self.to_cuda = to_cuda and torch.cuda.is_available()
        self.size_batch = None

        if to_cuda:
            self.operator.to(device=torch.device("cuda"), dtype=torch.complex64)
            self.adj_operator.to(device=torch.device("cuda"), dtype=torch.complex64)

            self.size_batch = size_batch

    def adjoint(
        self,
        batch_k_space: np.ndarray,
        batch_k_samples: np.ndarray,
        batch_sqrt_dcf: np.ndarray,
        coil_p: np.ndarray,
    ):
        """Single slice batch NUFFT operator. A wrapper for _adjoint method
        If gpu is used, this method calls _adjoint in a batched manner to avoid
        hitting GPU memory limitations.
        Args:
            batch_k_space(ndarray[complex64]): [Sb x NS x NC x NSp] k_space data
            batch_k_samples(ndarray[float]): [Sb x NS x NSp] star sampling information
            batch_sqrt_dcf(ndarray[complext]): [Sb x NS x NSp] square root of the dcf
            coil_p(ndarray[complex]): [NS x NS x NC] coil profile matrix
                [Sb,NS,NC,NSp] = [size_batch, num_sample, num_coil, num_spoke]
        Return:
            out_(torcharray[complex64]): [Sb x NC x NS x NS] image representation
        """
        if self.to_cuda:
            num_el = len(batch_k_space)
            xk = []
            for idx_batch in range(int(np.ceil(num_el / self.size_batch))):
                i_el = idx_batch * self.size_batch
                i_el_ = np.minimum((idx_batch + 1) * self.size_batch, num_el)

                xk.append(
                    self._adjoint(
                        copy(
                            torch.tensor(batch_k_space[i_el:i_el_]).to(torch.complex64)
                        ),
                        copy(batch_k_samples[i_el:i_el_]),
                        copy(batch_sqrt_dcf[i_el:i_el_]),
                        copy(coil_p),
                    )
                )
            xk = torch.cat(xk)
        else:
            xk = self._adjoint(
                torch.tensor(batch_k_space).to(torch.complex64),
                batch_k_samples,
                batch_sqrt_dcf,
                coil_p,
            )

        xk = np.array(xk.cpu())
        return xk

    def _adjoint(
        self,
        k_space: torch.tensor,
        k_samples: np.ndarray,
        sqrt_dcf: np.ndarray,
        coil_p: np.ndarray,
    ):
        """Single slice batch NUFFT operator.
        Args:
            k_space(torcharray[complex64]): [Sb x NS x NC x NSp] k_space data
            k_samples(ndarray[float]): [Sb x NS x NSp] star sampling information
            sqrt_dcf(ndarray[complext]): [Sb x NS x NSp] square root of the dcf
            coil_p(ndarray[complex]): [NS x NS x NC] coil profile matrix
                [Sb,NS,NC,NSp] = [size_batch, num_sample, num_coil, num_spoke]
        Return:
            out_(torcharray[complex64]): [Sb x NC x NS x NS] image representation
        """

        [_, _, n_spk] = sqrt_dcf.shape
        [dim_x_im, dim_y_im, _] = coil_p.shape

        # trajectory does not need to be a tensor as it is only a multiplicative factor
        k_traj = np.array(
            [
                np.array(
                    [np.real(el.ravel()) * 2 * np.pi, np.imag(el.ravel()) * 2 * np.pi]
                )
                for el in k_samples
            ]
        )
        k_traj = torch.tensor(k_traj).to(torch.float32)

        f_0 = torch.tensor(np.expand_dims(sqrt_dcf, 2)).to(torch.float32)
        f_1 = np.conj(np.expand_dims(np.transpose(coil_p, (2, 0, 1)), 0))
        f_1 = torch.tensor(f_1).to(torch.complex64)

        # TODO: Normalization constant to match previous implementations
        f_2 = np.sqrt(dim_x_im * dim_y_im) * np.pi / (2 * n_spk)
        f_3 = np.expand_dims(np.sum(np.abs(coil_p) ** 2, axis=2), (0, 1))
        f_norm = torch.tensor(f_2 * f_3).to(torch.float32)

        if self.to_cuda:
            k_space = k_space.to(device=torch.device("cuda"))
            k_traj = k_traj.to(device=torch.device("cuda"))
            f_0 = f_0.to(device=torch.device("cuda"), dtype=torch.complex64)
            f_1 = f_1.to(device=torch.device("cuda"), dtype=torch.complex64)
            f_norm = f_norm.to(device=torch.device("cuda"), dtype=torch.complex64)

        # adjust k for parallel adjoint calculation
        tmp_ = k_space * f_0
        tmp_ = tmp_.permute(1, 3, 0, 2)
        tmp_ = tmp_.reshape(-1, *tmp_.shape[-2:])
        tmp_ = tmp_.permute(1, 2, 0)

        out_ = self.adj_operator(tmp_, k_traj)
        out_ *= f_1

        out_ /= f_norm

        return out_

    def forward(
        self,
        batch_xk: np.ndarray,
        batch_k_samples: np.ndarray,
        batch_sqrt_dcf: np.ndarray,
        coil_p: np.ndarray,
    ):
        """Single slice batch NUFFT operator.
        Args:
            batch_xk(torcharray[complex64]): [Sb x NS x NS] image domain repr.
            batch_k_samples(ndarray[float]): [Sb x NS x NSp] star sampling
            batch_sqrt_dcf(ndarray[complext]): [Sb x NS x NSp] square root of the dcf
            coil_p(ndarray[complex]): [NS x NS x NC] coil profile matrix
                [Sb,NS,NC,NSp] = [s`ize_batch, num_sample, num_coil, num_spoke]
        Return:
            out_(torcharray[complex64]): [Sb x NS x NC x NSp] k_space data
        """
        if self.to_cuda:
            num_el = len(batch_xk)
            kk = []
            for idx_batch in range(int(np.ceil(num_el / self.size_batch))):
                i_el = idx_batch * self.size_batch
                i_el_ = np.minimum((idx_batch + 1) * self.size_batch, num_el)

                kk.append(
                    self._forward(
                        copy(torch.tensor(batch_xk[i_el:i_el_]).to(torch.complex64)),
                        copy(batch_k_samples[i_el:i_el_]),
                        copy(batch_sqrt_dcf[i_el:i_el_]),
                        copy(coil_p),
                    )
                )
            kk = torch.cat(kk)
        else:
            kk = self._forward(
                torch.tensor(batch_xk).to(torch.complex64),
                batch_k_samples,
                batch_sqrt_dcf,
                coil_p,
            )

        kk = np.array(kk.cpu())
        return kk

    def _forward(
        self,
        x: torch.tensor,
        k_samples: np.ndarray,
        sqrt_dcf: np.ndarray,
        coil_p: np.ndarray,
    ):
        """Single slice batch NUFFT operator.
        Args:
            x(torcharray[complex64]): [Sb x NS x NS] image domain representation
            k_samples(ndarray[float]): [Sb x NS x NSp] star sampling information
            sqrt_dcf(ndarray[complext]): [Sb x NS x NSp] square root of the dcf
            coil_p(ndarray[complex]): [NS x NS x NC] coil profile matrix
                [Sb,NS,NC,NSp] = [size_batch, num_sample, num_coil, num_spoke]
        Return:
            out_(torcharray[complex64]): [Sb x NS x NC x NSp] k_space data
        """

        [size_batch, n_sample, n_spk] = sqrt_dcf.shape
        [dim_x_im, dim_y_im, num_coil] = coil_p.shape

        # trajectory does not need to be a tensor as it is only a multiplicative factor
        k_traj = np.array(
            [
                np.array(
                    [np.real(el.ravel()) * 2 * np.pi, np.imag(el.ravel()) * 2 * np.pi]
                )
                for el in k_samples
            ]
        )
        k_traj = torch.tensor(k_traj).to(torch.float32)
        f_1 = np.expand_dims(np.transpose(coil_p, (2, 0, 1)), 0)
        f_1 = torch.tensor(f_1, dtype=torch.complex64)
        f_2 = torch.tensor(np.expand_dims(sqrt_dcf, 1)).to(torch.float32)
        f_3 = torch.tensor(np.sqrt(dim_x_im * dim_y_im)).to(torch.float32)

        smap_sz = (size_batch, num_coil, dim_x_im, dim_y_im)
        smap = torch.ones(*smap_sz, dtype=torch.complex64)

        if self.to_cuda:
            x = x.to(device=torch.device("cuda"))
            k_traj = k_traj.to(device=torch.device("cuda"))
            f_1 = f_1.to(device=torch.device("cuda"), dtype=torch.complex64)
            f_2 = f_2.to(device=torch.device("cuda"), dtype=torch.complex64)
            f_3 = f_3.to(device=torch.device("cuda"), dtype=torch.complex64)
            smap = smap.to(device=torch.device("cuda"))
            torch.cuda.synchronize()

        out_ = self.operator(x.unsqueeze(1) * f_1, k_traj, smaps=smap)

        out_ = out_.reshape(size_batch, num_coil, n_sample, n_spk)
        out_ *= f_2
        out_ /= f_3

        out_ = out_.permute(0, 2, 1, 3)

        return out_
