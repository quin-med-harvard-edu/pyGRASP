import os

from tqdm import tqdm

from load_convert.lc_helpers.read_write_helpers import (parse_info_csv,
                                                        parse_mru_csv,
                                                        update_info_csv,
                                                        update_loader)
from load_convert.load_vbvd import process_vbvd


def csv_to_dataset(
    path_csv: str,
    path_dataset: str,
    param_parser_grasp,
    param_parser_pproc,
    flag_reject_coil: bool = False,
    flag_compress_coil: bool = False,
    is_gpu: bool = False,
    num_spoke_coil: int = 1300,
):
    """Converts an MRU.csv file to a dataset format. Each element in the dataset
    includes k2, k3, coil profiles, k samples and dcf for the acquisition. This
    function also creates a loader.csv that is used in CRLMRUData class.
    Args:
        path_csv(str): full path to csv file
        path_dataset(str): path to dataset folder
        param_parser_grasp: json file for grasp parameters - will be edited in process_vbvd->load_single_vbvd
        param_parser_pproc: json file for postprocessing parameters - will be edited in process_vbvd->load_single_vbvd
        flag_reject_coil(bool): if True applies coil rejection
        flag_compress_coil(bool): if True applies PCA based coil compression
        is_gpu(bool): if True NUFFT operations are handled in gpu
        num_spoke_coil(int): number of spokes to estimate coil profile etc."""
    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)

    # Get information from mru data
    list_mru = parse_mru_csv(path_csv)
    list_mru_mrn = [list_mru[idx]["mrn"] for idx in range(len(list_mru))]

    path_info_csv = os.path.join(path_dataset, "info.csv")  # Information logfile
    path_loader_csv = os.path.join(path_dataset, "loader.csv")  # Data loader
    path_report_txt = os.path.join(
        path_dataset, "report.txt"
    )  # Report for user logfile
    path_devlog_txt = os.path.join(path_dataset, "devlog.txt")  # Developer logfile

    # Parse information csv to get already processed data
    if os.path.exists(path_info_csv):
        list_info = parse_info_csv(path_info_csv)
        list_info_mrn = [list_info[idx]["mrn"] for idx in range(len(list_info))]

        list_intersection = list(set(list_info_mrn).intersection(set(list_mru_mrn)))

        # Search for duplicate entries and remove them as well
        list_idx_intersection = []
        for el_intersect in list_intersection:
            list_idx_intersection.extend(
                [
                    idx_el_
                    for idx_el_, el_ in enumerate(list_mru_mrn)
                    if el_ == el_intersect
                ]
            )

        # Remove already existing data from list_mru
        for idx_intersection in list_idx_intersection:
            list_mru[idx_intersection] = None
        list_mru = list(filter(None, list_mru))
    else:
        list_info = []

    idx_sub = len(list_info) + 1
    with open(path_report_txt, "a+") as outfile:
        outfile.write("Data is taken from: {}\n".format(path_csv))
    with open(path_report_txt, "a") as outfile:
        outfile.write(
            "Params | reject:{}, compress:{}\n".format(
                flag_reject_coil, flag_compress_coil
            )
        )

    for el_subject in tqdm(list_mru):
        try:
            if len(el_subject["list_rec"]) == 0:
                raise FileNotFoundError

            k2, k3, dcf, k_samples, coil_p = process_vbvd(
                list_path_file=el_subject["list_rec"],
                param_parser_grasp=param_parser_grasp,
                param_parser_pproc=param_parser_pproc,
                num_spoke_coil=num_spoke_coil,
                flag_reject_coil=flag_reject_coil,
                flag_compress_coil=flag_compress_coil,
                is_gpu=is_gpu,
            )

            update_loader(
                path_loader_csv,
                path_dataset,
                idx_sub,
                {
                    "k22n": k2,
                    "k3n": k3,
                    "dcf": dcf,
                    "k_samples": k_samples,
                    "coilprofile": coil_p,
                },
            )

            with open(path_report_txt, "a") as outfile:
                outfile.write(
                    "Subject-{}: name:{}, mrn:{}\n".format(
                        idx_sub, el_subject["name"], el_subject["mrn"]
                    )
                )
            update_info_csv(path_info_csv, el_subject, idx_sub)
            idx_sub += 1
            del k2, k3, dcf, k_samples, coil_p

        except FileNotFoundError:
            with open(path_report_txt, "a") as outfile:
                outfile.write(
                    "Failed: File not found for name:{}, mrn:{}\n".format(
                        el_subject["name"], el_subject["mrn"]
                    )
                )

        except Exception as err:
            with open(path_devlog_txt, "a+") as outfile:
                outfile.write("{}".format(err))

    return 0
