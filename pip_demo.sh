inputOutput_path=inputOutput
mkdir -p $inputOutput_path

python convert_raw_to_dataset.py -i $inputOutput_path/MRU.csv -pg parameters/grasp_params.json -pp parameters/p_proc_params.json -o $inputOutput_path -fr True -fc True -nspkc 500

python demo_reconstruction.py -p parameters/grasp_params.json -d $inputOutput_path/loader.csv -o . -sid sub-1

python slice_to_vol.py -p parameters/p_proc_params.json -d $inputOutput_path/../sub-1/*/raw-rec/ -o $inputOutput_path/../sub-1/