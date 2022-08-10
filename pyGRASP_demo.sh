env_path="/environmentPath/"
code_path="/codePath/"
inputOutput_path="/inputOutputPath/"

"$env_path"/dce_env/bin/python3.7"" "$code_path"/dce_mri_grasp/convert_raw_to_dataset.py"" -i "$inputOutput_path"/MRU.csv"" -pg "$inputOutput_path"/grasp_params.json"" -pp "$inputOutput_path"/pproc.json"" -o $inputOutput_path -fr True -fc True -nspkc 500 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$env_path"/dce_env/lib""

"$env_path"/dce_env/bin/python3.7"" "$code_path"/dce_mri_grasp/demo_reconstruction.py"" -p "$inputOutput_path"/grasp_params.json"" -d "$inputOutput_path"/loader.csv"" -o . -sid sub-1

"$env_path"/dce_env/bin/python3.7"" "$code_path"/dce_mri_grasp/slice_to_vol.py"" -p "$inputOutput_path"/pproc.json"" -d "$inputOutput_path"/sub-1/*/raw-rec/"" -o "$inputOutput_path"/sub-1/""
