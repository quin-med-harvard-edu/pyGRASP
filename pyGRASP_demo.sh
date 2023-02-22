echo $"Available memory is: "$(awk '/^MemAvailable:/ { print $2; }' /proc/meminfo)$"kB"

env_path="yourEnvPath"
code_path="yourCodePath"
inputOutput_path="yourInputOutputPath"

$env_path/dce_env/bin/python3.7 $code_path/convert_raw_to_dataset.py -i $inputOutput_path/MRU.csv -pg $code_path/parameters/grasp_params.json -pp $code_path/parameters/p_proc_params.json -o $inputOutput_path -fr True -fc True -nspkc 500

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$env_path"/dce_env/lib

$env_path/dce_env/bin/python3.7 $code_path/demo_reconstruction.py -p $code_path/parameters/grasp_params.json -d $inputOutput_path/loader.csv -o $inputOutput_path/ -sid sub-1

$env_path/dce_env/bin/python3.7 $code_path/slice_to_vol.py -p $code_path/parameters/p_proc_params.json -d $inputOutput_path/sub-1/*/raw-rec/ -o $inputOutput_path/sub-1/