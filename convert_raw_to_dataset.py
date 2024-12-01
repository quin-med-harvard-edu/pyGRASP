# Developed by Aziz Kocanaogullari and QUIN Lab, 11/11/2021
""" This script utilizes a .csv file with specified format and converts it to a dataset
that is recognized for this repository.

The raw data file is expected to consist following columns:
Names(str): each row of names column indicate the name of the subject
Acquisition Date(int): YYYYMMDD integer indicating the date of acquisition
MRN(int): patient number
reconFile#(str): full path to reconsturction file
timePoints#(int): number of time points in the file (This is currently not used)

The aim of creating a dataset is also to anonymize the subject information. Instead
of personal information, each subject in the csv will be given a subject
identification number e.g. sub-#. This script creates a dataset folder to the provided
location --path_dataset with the following elements:
info.csv: a csv file that contains necessary information to identify each subject
loader.csv: data loader csv that includes paths to required slice data
{report,devlog}.txt: auxiliary files for developer
sub-#(folder): subject folder with k-space info, coil profiles, dcf, trajectory """

import argparse

from load_convert.convert_to_dataset import csv_to_dataset
from parameters.param_parse import GraspParamParse, PostProcessParamParse

parser = argparse.ArgumentParser(description="Arguments for Dataset Conversion")
parser.add_argument(
    "-i", "--path_csv", required=True, type=str, help="[path_csv] Path to mru-csv file"
)
parser.add_argument(
    "-pg",
    "--path_grasp_json",
    required=True,
    type=str,
    default="./parameters/grasp_params.json",
    help="[parameters] Path to grasp_params.json file",
)
parser.add_argument(
    "-pp",
    "--path_pproc_json",
    required=True,
    type=str,
    default="./parameters/p_proc_params.json",
    help="[parameters] Path to pproc.json file",
)
parser.add_argument(
    "-o",
    "--path_dataset",
    required=True,
    type=str,
    default=None,
    help="[path_dataset] Path to the dataset location",
)
parser.add_argument(
    "-igpu",
    "--is_gpu",
    required=False,
    type=bool,
    default=False,
    help="[is_gpu] if true set NUFFT to gpu",
)
parser.add_argument(
    "-fr",
    "--flag_reject_coil",
    required=False,
    type=bool,
    default=False,
    help="[flag_reject_coil] if true rejects outlier coils",
)
parser.add_argument(
    "-fc",
    "--flag_compress_coil",
    required=False,
    type=bool,
    default=False,
    help="[flag_compress_coil] if true compresses coils with PCA",
)
parser.add_argument(
    "-nspkc",
    "--num_spoke_coil",
    required=False,
    type=int,
    default=1300,
    help="[num_spoke_coil] number of spokes to estimate coil profiles",
)

args = parser.parse_args()
path_csv = args.path_csv
path_dataset = args.path_dataset
is_gpu = args.is_gpu
flag_reject_coil = args.flag_reject_coil
flag_compress_coil = args.flag_compress_coil
num_spoke_coil = args.num_spoke_coil

param_parser_grasp = GraspParamParse(
    json_path=args.path_grasp_json, csv_path=None, subject_id=None, save_path=None
)

param_parser_pproc = PostProcessParamParse(json_path=args.path_pproc_json)

csv_to_dataset(
    path_csv=path_csv,
    path_dataset=path_dataset,
    param_parser_grasp=param_parser_grasp,
    param_parser_pproc=param_parser_pproc,
    is_gpu=is_gpu,
    flag_reject_coil=flag_reject_coil,
    flag_compress_coil=flag_compress_coil,
    num_spoke_coil=num_spoke_coil,
)
