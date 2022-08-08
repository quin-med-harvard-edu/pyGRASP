import torch
import numpy as np
import csv
from torch.utils.data import DataLoader, Dataset
from copy import copy
import os
import hdf5storage as hdfs
from tqdm import tqdm


class CRLMRUData(Dataset):

    def __init__(self, csv_root: str,
                 type_parse: str = None):
        """ csv_root(str): root folder to dataloader csv file with path info
            type_parse(str): data parsing type
                -slice: treats each slice in a data independently """

        self.csv_root = csv_root
        list_data = []
        with open(self.csv_root, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                list_data.append(row)

        # Parse csv into desirable format
        # Check if row-length is cons`istent
        assert len(np.unique([len(row) for row in list_data])) == 1, \
            "The CSV file has unmatched rows!"

        self.primitive_info = copy(list_data[0])
        # Check necessary columns
        must_have_cols = ['id', 'k3n', 'k22n', 'coilprofile', 'k_samples',
                          'dcf', 'total_spokes']
        assert all([(it in self.primitive_info) for it in must_have_cols]), \
            "CSV file doesn't have the columns: {}".format(must_have_cols)

        self.primitive_dat = copy(list_data[1:])

        if type_parse == 'slice':
            self.data, self.info = self._parse_slice_level()
        else:
            print("Parsing:{} is not implemented keeping raw format".format(type_parse))
            self.data = copy(self.primitive_dat)
            self.info = copy(self.primitive_info)

        assert all([len(it_) == len(self.info) for it_ in self.data]), \
            "There's a parsing error in {}!".format(type_parse)

    def __len__(self):
        return len(self.primitive_dat)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        tmp_ = self.data[index]

        # Data shapes
        # k22n : N x num_ch x num_spoke x num_slice
        # k3n : N x num_ch x num_spoke x num_slice
        # coil_p : N x N x num_ch x num_slice
        # k_samples: N x num_spoke x num_slice
        # dcf: N x num_spoke x num_slice

        sample = {}
        for q_val_ in self.info:
            if q_val_ == 'id':
                sample[q_val_] = tmp_[self.info.index(q_val_)]
            elif q_val_ == 'total_spokes':
                sample[q_val_] = int(tmp_[self.info.index(q_val_)])
            else:
                tmp_mat_ = hdfs.loadmat(tmp_[self.info.index(q_val_)])
                tmp_mat_.pop('__header__', None)
                tmp_mat_.pop('__version__', None)
                tmp_mat_.pop('__globals__', None)

                assert (len(tmp_mat_.keys()) == 1), "Invalid data format!"
                sample[q_val_] = tmp_mat_[list(tmp_mat_.keys())[0]]

        return sample

    def _parse_slice_level(self):
        info_parse = ['id', 'k3n', 'k22n', 'coilprofile',
                      'k_samples', 'dcf', 'total_spokes']
        dat_parse = []

        root_dir = os.path.dirname(self.csv_root)

        for item_ in tqdm(self.primitive_dat):
            folder_k3n = os.path.join(root_dir, item_[self.primitive_info.index('k3n')])
            folder_k22n = os.path.join(root_dir,
                                       item_[self.primitive_info.index('k22n')])
            folder_cp = os.path.join(root_dir,
                                     item_[self.primitive_info.index('coilprofile')])
            folder_ks = os.path.join(root_dir,
                                     item_[self.primitive_info.index('k_samples')])
            folder_dcf = os.path.join(root_dir, item_[self.primitive_info.index('dcf')])
            total_spk = int(float(item_[self.primitive_info.index('total_spokes')]))

            for slice_idx, [file_k3n, file_k22n, file_cp, file_dcf, file_ks] in \
                    enumerate(zip(os.listdir(folder_k3n), os.listdir(folder_k22n),
                                  os.listdir(folder_cp), os.listdir(folder_dcf),
                                  os.listdir(folder_ks))):
                dat_parse.append(
                    [item_[self.primitive_info.index('id')] + '-s' + str(slice_idx),
                     os.path.join(folder_k3n, file_k3n),
                     os.path.join(folder_k22n, file_k22n),
                     os.path.join(folder_cp, file_cp),
                     os.path.join(folder_ks, file_ks),
                     os.path.join(folder_dcf, file_dcf),
                     total_spk])

        return dat_parse, info_parse
