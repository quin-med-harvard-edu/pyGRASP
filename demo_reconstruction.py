# Developed by Aziz Kocanaogullari and QUIN Lab, 02/04/2021
import numpy as np
from helpers.nufft.nufft_pt import PTNufft
from data_loader.crl_dataset import CRLMRUData
from helpers.pre_proces import down_sample_freq
from copy import copy
from rec_algo.temporal_sparse.grasp import GRASP
from datetime import datetime
import os
from parameters.param_parse import GraspParamParse
import scipy.io as sio
import time
import argparse

parser = argparse.ArgumentParser(description='Arguments for Grasp Reconstruction')
parser.add_argument('-p', '--root_json',
                    required=True,
                    type=str,
                    default='./parameters/grasp_params.json',
                    help='[parameters] Path to grasp_params.json file')
parser.add_argument('-d', '--root_csv',
                    required=True,
                    type=str,
                    default=None,
                    help='[domain] Path to the loader.csv file')
parser.add_argument('-o', '--save_root',
                    required=True,
                    type=str,
                    default=None,
                    help='[output] Path to the save location')
parser.add_argument('-sid', '--subject_id',
                    required=True,
                    type=str,
                    default='sub-1',
                    help='Subject id (check loader.csv 1st column)')
args = parser.parse_args()
param_parser = GraspParamParse(json_path=args.root_json,
                               csv_path=args.root_csv,
                               subject_id=args.subject_id,
                               save_path=args.save_root)

parameters = param_parser.get_param_val()

# General parameters
save_root = parameters['save_root']
root_csv_file = parameters['path_root_csv']
subject_id = parameters['subject_id']
spv = parameters['spv']
# Grasp parameters
grasp_lambda = parameters['lam']
max_num_iter = parameters['max_num_iter']
max_num_ls = parameters['max_num_ls']
# nufft parameters
nd = parameters['nd']
kd = parameters['kd']
jd = parameters['jd']
# if nufft has gpu option
is_gpu = parameters['is_gpu']
num_nufft_batch = parameters['size_nufft_batch']
# image resacling parameters
fov_factor = parameters['fov_scaling']

rec_save_folder = os.path.join(save_root, subject_id) + '/'
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

if not os.path.exists(rec_save_folder):
    os.makedirs(rec_save_folder)

crl_msu_dat = CRLMRUData(root_csv_file, type_parse='slice')
list_ids = [it_[crl_msu_dat.info.index('id')] for it_ in crl_msu_dat.data]
idx_sub = [list_ids.index(it_) for it_ in list_ids if subject_id + '-' in it_]

[num_sample, num_coil, num_spoke] = crl_msu_dat[idx_sub[0]]['k3n'].shape
num_slice = len(idx_sub)
num_vol = int(np.floor(num_spoke / spv))

# TODO: this is not the best way fix later
assert (num_sample / fov_factor) == nd[0], \
    "number of samples rescaled with field of view refactor does not match nd!"

path_folder = os.path.join(rec_save_folder, date_time)
path_raw_rec = os.path.join(path_folder, 'raw-rec')
if not os.path.exists(path_raw_rec):
    os.makedirs(path_raw_rec)

t = time.time()

for idx_s, dat_idx in enumerate(idx_sub):
    dat_slice = crl_msu_dat[dat_idx]
    print("Operating slice:{}/{}".format(idx_s + 1, num_slice))

    k_3 = np.expand_dims(dat_slice['k3n'] * np.power(10, 2), axis=0)
    k_samples = np.expand_dims(dat_slice['k_samples'], axis=0)
    sqrt_dcf = np.expand_dims(np.sqrt(dat_slice['dcf']), axis=0)
    k_3 = k_3 * np.expand_dims(sqrt_dcf, axis=2)

    k_3 = (dat_slice['k3n'] * np.power(10, 2))
    div_k_space = np.array([k_3[:, :, el * spv:(el + 1) * spv] for el in range(num_vol)])
    div_k_samples = np.array(
        [dat_slice['k_samples'][:, el * spv:(el + 1) * spv] for el in range(num_vol)])
    div_sqrt_dcf = np.array(
        [np.sqrt(dat_slice['dcf'])[:, el * spv:(el + 1) * spv] for el in range(num_vol)])
    coil_p = dat_slice['coilprofile'] / np.max(np.abs(dat_slice['coilprofile']))
    div_k_space = div_k_space * np.expand_dims(div_sqrt_dcf, axis=2)

    print("Computing NUFFT reconstructions!")
    div_k_space, div_k_samples, div_sqrt_dcf, coil_p = down_sample_freq(div_k_space,
                                                                        div_k_samples,
                                                                        div_sqrt_dcf,
                                                                        coil_p,
                                                                        fov_factor)
    nufft_obj = PTNufft(nd=nd, kd=kd, jd=jd,
                        to_cuda=is_gpu, size_batch=num_nufft_batch)

    xk = np.sum(nufft_obj.adjoint(div_k_space, div_k_samples, div_sqrt_dcf, coil_p),
                axis=1)
    print("Done!")

    grasp_obj = GRASP(k_samples=div_k_samples,
                      sqrt_dcf=div_sqrt_dcf,
                      coil_p=coil_p,
                      nufft_obj=nufft_obj,
                      lam=grasp_lambda * np.max(np.abs(xk)))
    xk_hat = copy(xk)

    if grasp_lambda != 0:
        for iter_grasp in range(3):
            xk_hat = grasp_obj.reconstruct(xk_hat, div_k_space,
                                           max_num_iter=max_num_iter,
                                           max_num_line_search=max_num_ls)

    _rec = {'rec': xk_hat}
    file_name = os.path.join(path_raw_rec, 's-' + str(idx_s) + '.mat')
    sio.savemat(file_name, _rec)

print("Total Time Elapsed:{}".format(time.time() - t))
param_file = os.path.join(path_folder, 'grasp_params.json')
param_parser.save_struct_to_file(param_file)
#
