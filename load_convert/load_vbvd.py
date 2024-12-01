import json
import os

import mapvbvd
import numpy as np
from helpers.nufft.nufft_pt import PTNufft
from pypilottone import radial
from tqdm import tqdm

from load_convert.ext.coils import calculate_csm_walsh
from load_convert.lc_helpers.calculate_dcf import calculate_dcf_traj
from load_convert.lc_helpers.coil_helpers import (compress_coil,
                                                  reject_coil_l2,
                                                  reject_coil_mi)


def process_vbvd(
    list_path_file: list,
    param_parser_grasp,
    param_parser_pproc,
    num_spoke_coil: int = 1300,
    flag_reject_coil: bool = False,
    flag_compress_coil: bool = False,
    is_gpu: bool = True,
    num_nufft_batch: int = 10,
):
    """Process vbvd data using a list of filenames from the same experiment.
    This also calculates coil profiles and applies respective coil operations.
        Args:
            list_path_file(list[str]): list of filenames
            num_spoke_coil(int): number of spokes to be used in coil estimation
                and compression operations.
            flag_reject_coil(bool): if True, rejects coils based on a likelihood
                measure (mutual information to average rec. by default)
            flag_compress_coil(bool): if True, compresses number of coils using PCA
            is_gpu(bool): non uniform fourier transform parameter if True sets to gpu
            num_nufft_batch(int): if is_gpu is True, determines the batch size
        Return:
            k2(ndarray[complex64]): Nsmp x Nc x Nspk x Nsli kspace data array
                Nsmp, Nc, Nspk, Nsli are number of samples, channels, spokes and
                slices respectively.
                , k3, dcf, k_samples, coil_p
            k3(ndarray[complex64]): Nsmp x Nc x Nspk x Nsli kspace data array
            dcf(ndarray[complex64]): Nsmp x Nspk compensation array
            k_samples(ndarray[complex64]): Nsmp x Nspk sampling trajectory array
            coil_p(ndarray[complex64]): Nsmp x Nsmp x Nc x Nsli coil profile array
    """
    # Load from path
    k2, k_samples, dcf = [], [], []
    list_coil = []
    for path_file in list_path_file:
        k2_, k_samples_, dcf_, num_chunk_, list_coil_ = load_single_vbvd(
            path_file=path_file,
            param_parser_grasp=param_parser_grasp,
            param_parser_pproc=param_parser_pproc,
        )
        k2.append(k2_)
        k_samples.append(k_samples_)
        dcf.append(dcf_)
        list_coil.append(list_coil_)
        # Release memory
        del k2_, k_samples_, dcf_

    if not bool(
        np.prod([list_coil[0] == list_coil[idx_] for idx_ in range(1, len(list_coil))])
    ):
        # TODO: Implement coil correction module
        raise NotImplementedError

    k2 = np.concatenate(k2, axis=2)
    k_samples = np.concatenate(k_samples, axis=-1)
    dcf = np.concatenate(dcf, axis=-1)

    # Convert k2 to k3
    k3 = np.fft.fft(k2, axis=-1).astype(np.complex64)

    [num_smp, num_ch, num_spoke, num_slice] = k3.shape
    nufft_obj = PTNufft(
        nd=(num_smp, num_smp),
        kd=(int(num_smp * 1.5), int(num_smp * 1.5)),
        jd=(6, 6),
        to_cuda=is_gpu,
        size_batch=num_nufft_batch,
    )

    if flag_reject_coil:
        idx_bad, _ = reject_coil_mi(
            k3[:, :, :num_spoke_coil, :],
            k_samples[:, :num_spoke_coil],
            dcf[:, :num_spoke_coil],
            nufft_obj=nufft_obj,
            reject_percent=0.6,
        )
        k3 = np.delete(k3, idx_bad, axis=1)

    if flag_compress_coil:
        k3 = compress_coil(k3, 0.01)

    [num_smp, num_ch, num_spoke, num_slice] = k3.shape

    # Get no coil reconstructions

    coil_p = []
    tqdm_bar = tqdm(range(num_slice), desc="computing coil profiles:", leave=True)
    for idx_slice in tqdm_bar:
        k_3 = k3[:, :, :num_spoke_coil, idx_slice] * np.expand_dims(
            np.sqrt(dcf[:, :num_spoke_coil]), axis=1
        )
        xk = nufft_obj.adjoint(
            np.expand_dims(k_3, axis=0),
            np.expand_dims(k_samples[:, :num_spoke_coil], axis=0),
            np.expand_dims(np.sqrt(dcf[:, :num_spoke_coil]), axis=0),
            np.ones([num_smp, num_smp, num_ch]),
        )
        coil_p_, pwr = calculate_csm_walsh(np.squeeze(xk))
        coil_p_ = (np.transpose(coil_p_, [1, 2, 0]) / np.max(np.abs(coil_p_))).astype(
            np.complex64
        )

        # if idx_slice == 0:
        #    coil_p_max = np.max(np.abs(coil_p_))

        # coil_p_ = (np.transpose(coil_p_, [1, 2, 0]) /
        #           coil_p_max).astype(np.complex64)
        coil_p.append(coil_p_)

    coil_p = np.array(coil_p)
    coil_p = np.transpose(coil_p, [1, 2, 3, 0])

    return k2, k3, dcf, k_samples, coil_p


def load_single_vbvd(path_file: str, param_parser_grasp, param_parser_pproc):
    """Loads vbvd file utilizing Siemens' provided module
    Args:
        path_file(str): absolute path to the data
    Return:
        k2(ndarray[complex64]): Nsmp x Nc x Nspk x Nsli kspace data array
            Nsmp, Nc, Nspk, Nsli are number of samples, channels, spokes and
            slices respectively. Slices are already decoupled with i-FFT.
        num_chunk(int): number of chunks in the data
        list_coil(list[str]): list of coil names that are used in the acquisition
    """
    file_stats = os.stat(path_file)
    print(f"File size is {file_stats.st_size/1024}kB")
    print(
        "Please monitor your memory to notice software failure due to insufficient memory. "
    )
    twix_obj = mapvbvd.mapVBVD(path_file)
    is_twix_obj_list = False

    if isinstance(twix_obj, list):
        twix_obj = twix_obj[1]
        is_twix_obj_list = True

    twix_obj.image.flagRemoveOS = False
    # twix_obj.image.flagRemoveOS = True
    twix_obj.image.squeeze = True
    k2_ = twix_obj.image[""]
    # implement here, this was manual OS removal
    # s = k2_.shape
    # img_space_OS_removed = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k2_, axes=(0)), axis=-1), axes=(0))
    # oversampling_indexes = np.arange(
    #     np.int46(k2_.shape[0] / 4), np.int64(k2_.shape[0] * 3 / 4)
    # )
    # kspace_justOS = np.fft.fft(
    #     np.fft.ifft(kspace_justOS, axis=0)[oversampling_indexes, ...], axis=0
    # )
    # ispace_zeroOS = np.fft.ifft(k2_, axis=0)
    # ispace_zeroOS[oversampling_indexes, ...] = np.complex64(0)
    # k2_ = np.fft.fft(ispace_zeroOS, axis=0)
    # del ispace_zeroOS

    # edit grasp and pproc parameters using values in data file
    edit_grasp_pproc_params(
        param_parser_grasp, param_parser_pproc, twix_obj, is_twix_obj_list
    )

    if len(k2_.shape) == 5:
        [num_sample, num_coil, num_spoke, num_slice, num_chunk] = k2_.shape
    else:
        [num_sample, num_coil, num_spoke, num_slice] = k2_.shape
        k2_ = np.expand_dims(k2_, axis=-1)
        num_chunk = 1

    # Extract scanner model from the header
    model_scanner = twix_obj.hdr.Dicom["ManufacturersModelName"]

    list_coil = []
    if model_scanner == "TrioTim" or model_scanner == "Avanto":
        for idx_c in range(num_coil):
            list_coil.append(
                str(
                    twix_obj.hdr.MeasYaps[
                        (
                            "asCoilSelectMeas",
                            "0",
                            "asList",
                            str(idx_c),
                            "sCoilElementID",
                            "tElement",
                        )
                    ][1:-1]
                )
            )

    elif (
        model_scanner == "Skyra"
        or model_scanner == "Prisma"
        or model_scanner == "MAGNETOM Vida"
        or model_scanner == "Prisma_fit"
    ):
        for idx_c in range(num_coil):
            list_coil.append(
                str(
                    twix_obj.hdr.MeasYaps[
                        (
                            "sCoilSelectMeas",
                            "aRxCoilSelectData",
                            "0",
                            "asList",
                            str(idx_c),
                            "sCoilElementID",
                            "tElement",
                        )
                    ][1:-1]
                )
            )
    else:
        print("Scanner: {} Not found".format(model_scanner))
        raise NotImplementedError

    # Chunk concatenation
    k2 = np.zeros(
        [num_sample, num_coil, num_spoke * num_chunk, num_slice], dtype=np.complex128
    )
    for idx in range(num_chunk):
        k2[:, :, idx * num_spoke : (idx + 1) * num_spoke, :] = k2_[:, :, :, :, idx]

    pilottone_on = True  # we'll want to add checks for this later
    if pilottone_on:
        print("extracting PT")
        k2, _ = radial.extract_pt_radial(k2, twix_obj.hdr)
        print("finished removing PT")

    # TODO: do not pass fixxed golden angle flag
    k_samples, dcf = calculate_dcf_traj(
        num_sample, num_spoke * num_chunk, num_chunk, flag_golden_angle=True
    )

    return k2, k_samples, dcf, num_chunk, list_coil


def edit_grasp_pproc_params(
    param_parser_grasp, param_parser_pproc, twix_obj, is_twix_obj_list
):
    try:
        image_shape = np.squeeze(
            twix_obj["image"][""]
        ).shape  # sz=size(squeeze(k{2}.image('')));
        hdr = twix_obj["hdr"]
        meas_yaps = hdr["MeasYaps"]  # FOVpar=k{2}.hdr.MeasYaps.sSliceArray.asSlice{1};
        no_slices = hdr["Config"][
            "NoImagesPerSlab"
        ]  # noSlices=k{2}.hdr.Config.NoImagesPerSlab;

        nx_image = image_shape[0] / 2  # Nx_image = sz(1) / 2;
        ny_image = image_shape[0] / 2  # Ny_image = sz(1) / 2;

        dx = (
            meas_yaps["sSliceArray", "asSlice", "0", "dReadoutFOV"] / nx_image
        )  # dx = FOVpar.dReadoutFOV / (sz(1) / 2);
        dy = (
            meas_yaps["sSliceArray", "asSlice", "0", "dPhaseFOV"] / ny_image
        )  # dy = FOVpar.dPhaseFOV / (sz(1) / 2);
        dz = (
            meas_yaps["sSliceArray", "asSlice", "0", "dThickness"] / no_slices
        )  # dz = FOVpar.dThickness / noSlices;

        tot_scan_time = meas_yaps[
            "lTotalScanTimeSec",
        ]  # totScanTime = k{2}.hdr.MeasYaps.lTotalScanTimeSec;
        spv = param_parser_grasp.params_struct["spv"]["val"]
        if is_twix_obj_list:
            slice_os = hdr["Dicom"]["flSliceOS"]  # sliceOS=k{2}.hdr.Dicom.flSliceOS
            # num_vols = image_shape[2] // spv                                      # numVols = floor(sz(3) / 34);
        else:
            slice_os = meas_yaps[
                "sKSpace", "dSliceOversamplingForDialog"
            ]  # sliceOS=k.hdr.MeasYaps.sKSpace.dSliceOversamplingForDialog;
            # num_vols = (image_shape[2] * image_shape[4]) // spv                   # numVols=floor(sz(3)*sz(5)/34);

        if len(image_shape) == 5:
            temp_res = (
                spv * tot_scan_time / (image_shape[2] * image_shape[4])
            )  # tempRes=34*totScanTime/(sz(3)*sz(5));
        elif len(image_shape) == 4:
            temp_res = (
                spv * tot_scan_time / image_shape[2]
            )  # tempRes = 34 * totScanTime / sz(3);
        else:
            temp_res = 0

        # print(param_parser_grasp.file_path)
        param_parser_grasp.params_struct["nd"]["val"] = [2.0 * nx_image, 2.0 * ny_image]
        param_parser_grasp.params_struct["kd"]["val"] = [3.0 * nx_image, 3.0 * ny_image]
        param_parser_grasp.params_struct["fov_scaling"]["val"] = 1
        param_parser_grasp.save_struct_to_file(param_parser_grasp.file_path)

        param_parser_pproc.params_struct["post_img_size"]["val"] = [
            nx_image,
            ny_image,
            no_slices,
        ]
        if not slice_os:
            slice_os = 0.0
        param_parser_pproc.params_struct["rate_os"]["val"] = 1.0 + slice_os
        dx_dict = {
            "section": "slice_resampling_params",
            "helpTip": "dx",
            "val": dx,
            "type": "float",
        }
        print(f"DX DICT: {dx_dict}")
        dy_dict = {
            "section": "slice_resampling_params",
            "helpTip": "dy",
            "val": dy,
            "type": "float",
        }
        dz_dict = {
            "section": "slice_resampling_params",
            "helpTip": "dz",
            "val": dz,
            "type": "float",
        }
        dt_dict = {
            "section": "slice_resampling_params",
            "helpTip": "dt",
            "val": temp_res,
            "type": "float",
        }
        print(f"DT DICT: {dt_dict}")
        param_parser_pproc.params_struct["dx"] = dx_dict
        param_parser_pproc.params_struct["dy"] = dy_dict
        param_parser_pproc.params_struct["dz"] = dz_dict
        param_parser_pproc.params_struct["dt"] = dt_dict
        param_parser_pproc.save_struct_to_file(param_parser_pproc.file_path)
        # "flag_fft_shift":
        # {
        #  "section":"slice_resampling_params",
        #  "helpTip":"if true applies fft-shift along slice axis",
        #  "val":true,
        #  "type":"bool"
        # }
        #
    except Exception as e:
        print(f"An exception occurred in edit_grasp_params: {e}")
