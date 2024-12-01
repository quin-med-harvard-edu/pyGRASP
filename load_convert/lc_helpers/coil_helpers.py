import numpy as np
from helpers.nufft.base_nufft import NUFFTObject
from helpers.pre_proces import divide_in_time
from joblib import Parallel, delayed
from tqdm import tqdm


def compress_coil(x: np.ndarray, ev_tol: float = 0.05):
    """Compresses along coil axis with complex PCA
    Args:
        x(ndarray[complex]): [num_sample x num_channel x num_spoke x num_slice]
            k_space data that is going to be compressed
        ev_tol(float): eigenvalue tolerance in percentage. The lower eigenvalues
            are removed from projection.
    Return:
        map_x(ndarray[complex]): [num_sample x less_channel x num_spoke x num_slice]
            compressed k_space data"""

    [num_smp, num_ch, num_spoke, num_slice] = x.shape

    # Adjust for 16 gig memory
    percent_ = np.minimum(1, 17179869184 / (x.size * x.itemsize))
    num_spoke_prime = int(percent_ * num_spoke)
    dat_ = np.transpose(x[:, :, :num_spoke_prime, :], [0, 2, 3, 1])
    dat_ = np.reshape(dat_, [num_smp * num_spoke_prime * num_slice, num_ch])
    dat_ = dat_ - np.mean(dat_, axis=0)
    cov_ = np.cov(dat_.transpose())
    del dat_
    s, v = np.linalg.eig(cov_)
    s = np.real(s)
    num_cmp = np.sum(s > (s[0] * ev_tol))
    projection_matrix = (v.T[:][:num_cmp]).T

    dat_ = np.transpose(x[:, :, :, :], [0, 2, 3, 1])
    dat_ = np.reshape(dat_, [num_smp * num_spoke * num_slice, num_ch])
    dat_ = dat_.dot(projection_matrix)
    dat_ = np.reshape(dat_, [num_smp, num_spoke, num_slice, num_cmp])
    dat_ = np.transpose(dat_, [0, 3, 1, 2])
    return dat_


def image_mi(x: np.ndarray, y: np.ndarray, bins: tuple = (50, 50)):
    """Calculates mutual inforation between two images.
    Args:
        x(ndarray[]): Nsmp x Nsmp image
        y(ndarray[]): Nsmp x Nsmp image
        bins(tuple[int,int]): number of bins for the joint pdf (histogram)
    Return:
        mi(float): mutual information between two images
    """
    joint = np.histogram2d(x.ravel(), y.ravel(), bins=bins)[0] + np.power(0.1, 30)
    joint /= np.sum(joint)

    mar_x = np.sum(joint, axis=0).reshape((-1, joint.shape[0]))
    mar_y = np.sum(joint, axis=1).reshape((joint.shape[1], -1))

    mi = (
        np.sum(joint * np.log2(joint))
        - np.sum(mar_x * np.log2(mar_x))
        - np.sum(mar_y * np.log2(mar_y))
    )
    return mi


def compute_coil_mi(
    k3: np.ndarray,
    k_samples: np.ndarray,
    sqrt_dcf: np.ndarray,
    coil_p: np.ndarray,
    nufft_obj: NUFFTObject,
    num_max_spoke: int = 1500,
    spv: int = 34,
):
    """Calculates the mutual information (MI) for each image to the average image.
    The higher MI, the higher contribution to the reconstruction is.
    Removes the coils with low MI.
    Args:
        k3(ndarray[complex]): num_sample x num_channel x num_spk x num_slice
            k space data. This is converted to image domain for score calc.
        k_samples(ndarray[complex]): num_sample x num_spk k space trajectory
        sqrt_dcf(ndarray[complex]): num_sample x num_spk compensation function
        coil_p(ndarray[complex]): num_sample x num_sample x num_channel coil profiles
        nufft_obj(NUFFTObject): non uniform Fourier transform for the data
        num_max_spoke(int): number of maximum spokes used
        spv(int): spokes per volume
    Return:
        idx_reject(ndarray(int)): rejected indices
        coil_mi(ndarray[float]): num_channel x 1 mutual information value vector."""

    ref_coil = np.squeeze(
        nufft_obj.adjoint(
            np.expand_dims(k3 / np.expand_dims(sqrt_dcf, axis=1), axis=0),
            np.expand_dims(k_samples, axis=0),
            np.expand_dims(sqrt_dcf, axis=0),
            coil_p,
        )
    )
    x_ref = np.squeeze(np.sum(ref_coil, axis=0))

    div_k_space, div_k_samples, div_sqrt_dcf = divide_in_time(
        k3, k_samples, sqrt_dcf, num_max_spoke, spv
    )
    del k3, k_samples, sqrt_dcf
    rec_slice = nufft_obj.adjoint(
        div_k_space / np.expand_dims(div_sqrt_dcf, axis=2),
        div_k_samples,
        div_sqrt_dcf,
        coil_p,
    )
    print("Done!")

    [num_vol, num_ch, num_smp, _] = rec_slice.shape

    mi_slice = []
    for idx_ch in range(num_ch):
        mi_slice.append(
            np.sum(
                Parallel(n_jobs=-2)(
                    delayed(image_mi)(
                        np.squeeze(rec_slice[idx_vol, idx_ch, :, :]),
                        x_ref,
                        bins=(num_smp, num_smp),
                    )
                    for idx_vol in range(num_vol)
                )
            )
        )
    mi_slice = np.array(mi_slice)

    return mi_slice


def reject_coil_mi(
    k3: np.ndarray,
    k_samples: np.ndarray,
    dcf: np.ndarray,
    nufft_obj: NUFFTObject,
    min_num_coil: int = 8,
    reject_percent: float = 0.2,
):
    """Calculates the mutual information (MI) for each image to the average image.
    The higher MI, the higher contribution to the reconstruction is.
    Removes the coils with low MI.
    Args:
        k3(ndarray[complex]): num_sample x num_channel x num_spk x num_slice
            k space data. This is converted to image domain for score calc.
        k_samples(ndarray[complex]): num_sample x num_spk k space trajectory
        dcf(ndarray[complex]): num_sample x num_spk compensation function
        nufft_obj(NUFFTObject): non uniform Fourier transform for the data
        min_num_coil(int): minimum number of coils allowed
        reject_percent(float): percentage of coils to be removed
    Return:
        idx_reject(ndarray(int)): rejected indices
        coil_mi(ndarray[float]): num_channel x 1 mutual information value vector."""
    reject_percent = np.maximum(np.minimum(reject_percent, 1), 0)

    [num_smp, num_ch, num_spoke, num_slice] = k3.shape
    tqdm_bar = tqdm(range(num_slice), desc="computing mi score:", leave=True)
    mi = []
    for idx_slice in tqdm_bar:
        div_k_space, div_k_samples, div_sqrt_dcf = divide_in_time(
            np.squeeze(k3[:, :, :, idx_slice]), k_samples[:, :], np.sqrt(dcf[:, :])
        )
        num_vol = div_k_space.shape[0]
        div_k_space = div_k_space * np.expand_dims(div_sqrt_dcf, axis=2)
        xk = nufft_obj.adjoint(
            div_k_space,
            div_k_samples,
            div_sqrt_dcf,
            np.ones([num_smp, num_smp, num_ch]),
        )

        x_ref = np.squeeze(
            np.sum(
                nufft_obj.adjoint(
                    np.expand_dims(
                        k3[:, :, :, idx_slice] * np.expand_dims(np.sqrt(dcf), axis=1),
                        axis=0,
                    ),
                    np.expand_dims(k_samples, axis=0),
                    np.expand_dims(np.sqrt(dcf), axis=0),
                    np.ones([num_smp, num_smp, num_ch]),
                ),
                axis=1,
            )
        )
        mi_slice = []
        for idx_ch in range(num_ch):
            mi_slice.append(
                np.sum(
                    Parallel(n_jobs=-1)(
                        delayed(image_mi)(
                            np.squeeze(xk[idx_vol, idx_ch, :, :]),
                            np.squeeze(x_ref),
                            bins=(num_smp, num_smp),
                        )
                        for idx_vol in range(num_vol)
                    )
                )
            )
        mi.append(mi_slice)
        del xk, x_ref, div_k_samples, div_sqrt_dcf

    mi = np.array(mi)
    cnt_mi = np.mean(mi, axis=0)
    idx_reject = np.argsort(cnt_mi)[
        : int(np.minimum(num_ch * reject_percent, np.maximum(num_ch - min_num_coil, 0)))
    ]

    return idx_reject, cnt_mi


def reject_coil_l2(
    k3: np.ndarray,
    k_samples: np.ndarray,
    dcf: np.ndarray,
    nufft_obj: NUFFTObject,
    min_num_coil: int = 8,
    reject_percent: float = 0.2,
):
    """Calculates the mutual information (MI) for each image to the average image.
    The higher MI, the higher contribution to the reconstruction is.
    Removes the coils with low MI.
    Args:
        k3(ndarray[complex]): num_sample x num_channel x num_spk x num_slice
            k space data. This is converted to image domain for score calc.
        k_samples(ndarray[complex]): num_sample x num_spk k space trajectory
        dcf(ndarray[complex]): num_sample x num_spk compensation function
        nufft_obj(NUFFTObject): non uniform Fourier transform for the data
        min_num_coil(int): minimum number of coils allowed
        reject_percent(float): percentage of coils to be removed
    Return:
        idx_reject(ndarray(int)): rejected indices
        cnt_l2(ndarray[float]): num_channel x 1 negative l2 value vector."""

    reject_percent = np.maximum(np.minimum(reject_percent, 1), 0)

    [num_smp, num_ch, num_spoke, num_slice] = k3.shape
    l2 = []
    tqdm_bar = tqdm(range(num_slice), desc="computing mi score:", leave=True)
    for idx_slice in tqdm_bar:
        div_k_space, div_k_samples, div_sqrt_dcf = divide_in_time(
            np.squeeze(k3[:, :, :, idx_slice]), k_samples[:, :], np.sqrt(dcf[:, :])
        )
        num_vol = div_k_space.shape[0]
        div_k_space = div_k_space * np.expand_dims(div_sqrt_dcf, axis=2)
        xk = nufft_obj.adjoint(
            div_k_space,
            div_k_samples,
            div_sqrt_dcf,
            np.ones([num_smp, num_smp, num_ch]),
        )

        x_ref = nufft_obj.adjoint(
            np.expand_dims(
                k3[:, :, :, idx_slice] * np.expand_dims(np.sqrt(dcf), axis=1), axis=0
            ),
            np.expand_dims(k_samples, axis=0),
            np.expand_dims(np.sqrt(dcf), axis=0),
            np.ones([num_smp, num_smp, num_ch]),
        )
        l2_slice = []
        for idx_ch in range(num_ch):
            l2_slice.append(
                np.linalg.norm(xk[:, idx_ch, :, :] - x_ref[:, idx_ch, :, :])
            )
        l2.append(l2_slice)
        del xk, x_ref, div_k_samples, div_sqrt_dcf

    l2 = np.array(l2)
    cnt_l2 = -np.mean(l2, axis=0)
    idx_reject = np.argsort(cnt_l2)[
        : int(np.minimum(num_ch * reject_percent, np.maximum(num_ch - min_num_coil, 0)))
    ]

    return idx_reject, cnt_l2


def reject_coil_correlation(
    x: np.ndarray,
    dim_cov: tuple = (10, 10),
    w: tuple = (20, 20),
    min_num_coil: int = 8,
    reject_percent: float = 0.2,
):
    [num_slice, num_ch, num_smp, _] = x.shape

    cov_error = np.zeros([num_slice, num_ch])

    for idx_ch in tqdm(range(num_ch)):
        for idx_slice in range(num_slice):
            corr_cov = np.mean(
                np.mean(
                    [
                        [
                            np.corrcoef(
                                x[
                                    idx_slice,
                                    idx_ch,
                                    idx_x : idx_x + dim_cov[0],
                                    idx_y : idx_y + dim_cov[1],
                                ]
                            )
                            for idx_x in range(w[0] - dim_cov[0])
                        ]
                        for idx_y in range(w[1] - dim_cov[1])
                    ],
                    axis=0,
                ),
                axis=0,
            )

            corr_cov += np.mean(
                np.mean(
                    [
                        [
                            np.corrcoef(
                                x[
                                    idx_slice,
                                    idx_ch,
                                    idx_x : idx_x + dim_cov[0],
                                    idx_y : idx_y + dim_cov[1],
                                ]
                            )
                            for idx_x in range(w[0] - dim_cov[0])
                        ]
                        for idx_y in range(num_smp - w[1], num_smp - dim_cov[1])
                    ],
                    axis=0,
                ),
                axis=0,
            )

            corr_cov += np.mean(
                np.mean(
                    [
                        [
                            np.corrcoef(
                                x[
                                    idx_slice,
                                    idx_ch,
                                    idx_x : idx_x + dim_cov[0],
                                    idx_y : idx_y + dim_cov[1],
                                ]
                            )
                            for idx_x in range(num_smp - w[0], num_smp - dim_cov[0])
                        ]
                        for idx_y in range(w[1] - dim_cov[1])
                    ],
                    axis=0,
                ),
                axis=0,
            )

            corr_cov += np.mean(
                np.mean(
                    [
                        [
                            np.corrcoef(
                                x[
                                    idx_slice,
                                    idx_ch,
                                    idx_x : idx_x + dim_cov[0],
                                    idx_y : idx_y + dim_cov[1],
                                ]
                            )
                            for idx_x in range(num_smp - w[0], num_smp - dim_cov[0])
                        ]
                        for idx_y in range(num_smp - w[1], num_smp - dim_cov[1])
                    ],
                    axis=0,
                ),
                axis=0,
            )
            corr_cov /= 4

            corr_cov /= np.mean(np.diag(corr_cov))
            cov_error[idx_slice, idx_ch] = np.linalg.norm(
                corr_cov - np.eye(corr_cov.shape[0])
            )

    idx_reject = np.argsort(np.mean(np.abs(cov_error), axis=0))[
        : int(np.maximum(num_ch * reject_percent, min_num_coil))
    ]

    return idx_reject, cov_error
