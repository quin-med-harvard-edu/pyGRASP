# Developed by Aziz Kocanaogullari and QUIN Lab, 02/04/2021
import argparse
import os
from datetime import datetime

import nibabel as nib
import numpy as np
import scipy.io as sio

from helpers.post_process import interpolate_slice
from parameters.param_parse import PostProcessParamParse

def corrHead(fileName, dx, dy, dz,flag_interpolate_z=False):
    imo = nib.load(fileName)
    if flag_interpolate_z:
        newheader = imo.header.copy()
        newheader['pixdim'][1] = dx
        newheader['pixdim'][2] = dy
        newheader['pixdim'][3] = dz
        newheader['pixdim'][4] = dt

        #newheader['qoffset_x'] = -155.138
        #newheader['qoffset_y'] = 13.7454
        #newheader['qoffset_z'] = 218.226
    else: 
        newheader = imo.header
    im = imo.get_fdata()
    imonew = nib.Nifti1Image(im, affine=imo.affine, header=newheader)
    nib.save(imonew, fileName)


def corrAff(fileName):
    imo = nib.load(fileName)
    newaffine = imo.affine
    newaffine[0, :] = [-1, 0, 0, 0]
    newaffine[1, :] = [0, 0, -1, 0]
    newaffine[2, :] = [0, -1, 0, 0]
    newaffine[3, :] = [0, 0, 0, 1]
    im = imo.get_fdata()
    imonew = nib.Nifti1Image(im, affine=newaffine, header=imo.header)
    nib.save(imonew, fileName)

parser = argparse.ArgumentParser(description='Arguments for Post Processing')
parser.add_argument('-p', '--root_json',
                    required=True,
                    type=str,
                    default='./parameters/p_proc_params.json',
                    help='[parameters] Path to grasp_params.json file')
parser.add_argument('-d', '--root_rec',
                    required=True,
                    type=str,
                    default=None,
                    help='[domain] Path that that includes raw_rec')

parser.add_argument('-o', '--root_out',
                    required=True,
                    type=str,
                    default=None,
                    help='[output] Save path')
parser.add_argument('--donotscale',
                    action="store_true",
                    help='scale images to nifti format with max val of 32000')
parser.add_argument('--no_interpolate_z',
                    action="store_true",
                    help='do not interpolate slice')
parser.add_argument('--no_fft_shift',
                    action="store_true",
                    help='do not fft shift the slices')




args = parser.parse_args()

param_parser = PostProcessParamParse(json_path=args.root_json)
parameters = param_parser.get_param_val()

# flags 
scaleto32000=False if args.donotscale else True
flag_interpolate_z=False if args.no_interpolate_z else True
flag_fft_shift =False if args.no_fft_shift else True 


# General parameters
root_rec = args.root_rec
rec_save_folder = args.root_out
# Post Processing Parameters
post_img_size = parameters["post_img_size"]
rate_os = parameters["rate_os"]
flag_fft_shift = parameters["flag_fft_shift"]
dx = parameters["dx"]
dy = parameters["dy"]
dz = parameters["dz"]
dt = parameters["dt"]

folder_raw_rec = root_rec
# folder_raw_rec = os.path.join(root_rec, 'raw-rec')
# assert os.path.exists(folder_raw_rec), "No raw files detected at:{}".format(
#     folder_raw_rec)

rec_volume = []
mat_files = sorted(
    [f for f in os.listdir(folder_raw_rec) if f.endswith(".mat")],
    key=lambda x: int(x.split("-")[1].split(".")[0]),
)

for rec_file in mat_files:
    if rec_file.endswith(".mat"):
        rec_volume.append(
            np.absolute(sio.loadmat(os.path.join(folder_raw_rec, rec_file))["rec"])
        )
    # rec_volume.append(sio.loadmat(os.path.join(folder_raw_rec, rec_file))['rec'])
rec_volume = np.array(rec_volume)

init_num_slice = rec_volume.shape[0]
# Interpolate slices to match previous reconstructions
rec_volume = interpolate_slice(rec_volume, post_img_size, rate_os, flag_fft_shift=flag_fft_shift,flag_interpolate_z=flag_interpolate_z)

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

print("Saving...")
# To save in nifti format the data should be uint16
max_p = np.max(np.abs(rec_volume))
min_p = np.min(np.abs(rec_volume))
scaler = 2**15 if scaleto32000 else 1 
rec_vol_16 = np.array(np.round((np.abs(rec_volume) - min_p) /
                               (max_p - min_p) * scaler), dtype=np.uint16)
# save 4D volume
# Change dimensions from NSl x Nv x Ns x Ns to Ns x Ns x NSl x Nv
rec_vol_16 = np.transpose(rec_vol_16, [2, 3, 0, 1])
file_name = os.path.join(rec_save_folder, date_time + "-rec4D.nii.gz")
nii_dat = nib.Nifti1Image(rec_vol_16, np.eye(4))

# nii_dat.header['pixdim'][1] = dx #1.33929
# nii_dat.header['pixdim'][2] = dy #1.33929
# nii_dat.header['pixdim'][3] = dz #3.50018 * init_num_slice / post_img_size[0]

nii_dat.affine[0, :] = [-1, 0, 0, 0]
nii_dat.affine[1, :] = [0, 0, -1, 0]
nii_dat.affine[2, :] = [0, -1, 0, 0]
nii_dat.affine[3, :] = [0, 0, 0, 1]

if not os.path.exists(rec_save_folder):
    os.makedirs(rec_save_folder)

nii_dat.to_filename(file_name)

corrHead(file_name, dx, dy, dz,flag_interpolate_z=flag_interpolate_z)

param_file = os.path.join(rec_save_folder, date_time + "-params.json")
param_parser.save_struct_to_file(param_file)
