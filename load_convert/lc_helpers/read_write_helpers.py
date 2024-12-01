import csv
import os

import scipy.io as sio
from tqdm import tqdm


def parse_mru_csv(path_csv: str):
    with open(path_csv, mode="r") as infile:
        reader = csv.reader(infile)
        rows = [row for row in reader]

    names_col = rows[0]
    idx_name = names_col.index("Names")
    idx_mrn = names_col.index("MRN")
    idx_recon = [names_col.index(i) for i in names_col if "recon" in i]

    list_subject = []
    for idx in range(1, len(rows)):
        list_recon = []
        for i_recon in idx_recon:
            if rows[idx][i_recon].strip():
                list_recon.append(rows[idx][i_recon])

        list_subject.append(
            {
                "name": rows[idx][idx_name],
                "mrn": rows[idx][idx_mrn],
                "list_rec": list_recon,
            }
        )

    return list_subject


def parse_info_csv(path_csv: str):
    with open(path_csv, mode="r") as infile:
        reader = csv.reader(infile)
        rows = [row for row in reader]

    names_col = rows[0]
    idx_name = names_col.index("name")
    idx_mrn = names_col.index("mrn")
    idx_id = names_col.index("sub-id")

    list_subject = []
    for idx in range(1, len(rows)):
        list_subject.append(
            {
                "name": rows[idx][idx_name],
                "mrn": rows[idx][idx_mrn],
                "sub-id": rows[idx][idx_id],
            }
        )

    return list_subject


def update_info_csv(path_csv: str, update_el: dict, idx_sub: int):
    field_names = [None, None, None]
    if os.path.exists(path_csv):
        with open(path_csv, mode="r") as infile:
            reader = csv.reader(infile)
            rows = [row for row in reader]
        names_col = rows[0]
        idx_name = names_col.index("name")
        idx_mrn = names_col.index("mrn")
        idx_id = names_col.index("sub-id")
    else:
        idx_name, idx_mrn, idx_id = 0, 1, 2
        with open(path_csv, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["name", "mrn", "sub-id"])

    field_names[idx_name], field_names[idx_mrn], field_names[idx_id] = (
        "name",
        "mrn",
        "sub-id",
    )

    with open(path_csv, "a", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=field_names)
        writer.writerow(
            {
                "name": update_el["name"],
                "mrn": update_el["mrn"],
                "sub-id": "sub-{}".format(str(idx_sub)),
            }
        )
    return True


def update_loader(path_csv: str, path_dataset: str, idx_sub: int, dat: dict):
    field_names = [
        "id",
        "k22n",
        "k3n",
        "coilprofile",
        "k_samples",
        "dcf",
        "total_spokes",
    ]
    if not os.path.exists(path_csv):
        with open(path_csv, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(field_names)

    list_keys = list(dat.keys())
    [num_smp, num_ch, num_spoke, num_slice] = dat["k3n"].shape

    csv_dat = {"id": "sub-{}".format(idx_sub)}

    save_root = os.path.join(path_dataset, "sub_{}".format(idx_sub))
    for key_ in list_keys:
        save_key_ = os.path.join(save_root, key_)
        if not os.path.exists(save_key_):
            os.makedirs(save_key_)
        csv_dat = {**csv_dat, **{key_: save_key_}}
        for idx_slice in tqdm(range(num_slice)):
            if key_ == "k3n" or key_ == "k22n" or key_ == "coilprofile":
                dat_save = dat[key_][:, :, :, idx_slice]
            else:
                dat_save = dat[key_]
            if idx_slice < 10:
                save_name = "{}_slice_0{}".format(key_, idx_slice)
            else:
                save_name = "{}_slice_{}".format(key_, idx_slice)
            sio.savemat(
                os.path.join(save_key_, save_name) + ".mat", {save_name: dat_save}
            )
            del dat_save

    csv_dat["total_spokes"] = num_spoke
    with open(path_csv, "a", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=field_names)
        writer.writerow(csv_dat)

    return True
