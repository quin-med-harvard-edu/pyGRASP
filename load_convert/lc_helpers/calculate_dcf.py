import numpy as np


def calculate_dcf_traj(
    res_base: int,
    num_spoke: int,
    num_chunk: int,
    flag_golden_angle: bool = True,
    offset_spokes: int = 0,
    corr_grad: float = 0.4,
):
    """Calculates density compensation function and radial angles
    Args:
         res_base(int): base resolution for the signal (number of samples in kspace)
         num_spoke(int): number of total spokes in the kspace
         num_chunk(int): number of chunks
         flag_golden_angle(bool): this function can work both for golden angle and
            radial acquisition. if true returns for golden angle.
        offset_spokes(int): number of spokes to be removed
        corr_grad(float): during acquisition, lines usually do not cross through
            the origin. This corrects the error
    Return:
        k(ndarray[complex]): k samples for reconstruction
        dcf(ndarray[complex]): dcf for reconstruction
    """

    # Prepare the density compensation function (DCF)
    dcf_row = (
        np.abs(res_base * 0.5 - (np.arange(res_base) + 1 - 0.5))
        * np.pi
        / (num_spoke / num_chunk)
    )
    dcf = np.repeat(np.expand_dims(dcf_row, -1), num_spoke, axis=1)

    if flag_golden_angle:
        ga = 111.246117975 / 180 * np.pi
        phi = np.arange(
            start=np.pi / 2 + offset_spokes * ga,
            stop=ga * num_spoke / num_chunk + offset_spokes * ga,
            step=ga,
        )
    else:
        phi = np.zeros(1, num_spoke / num_chunk)
        for idx in range(num_spoke / num_chunk):
            phi[1, idx] = np.pi / num_spoke * num_chunk * idx
            if np.mod(idx, 2) == 1:
                phi[1, idx] = phi[1, idx] + np.pi

    phi = np.mod(phi, 2 * np.pi)

    rho = np.linspace(0, res_base - 1, res_base) - (res_base - corr_grad) / 2
    rho /= res_base

    k = np.dot(np.expand_dims(rho, axis=-1), np.expand_dims(np.exp(-1j * phi), axis=0))
    k = np.tile(k, (1, num_chunk))

    return k, dcf
