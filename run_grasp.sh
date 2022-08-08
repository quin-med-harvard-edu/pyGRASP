#!/bin/bash
p1=$1
p2=$2
p3=$3
p4=$4

python convert_raw_to_dataset.py -i /inputfolder/subject.csv -pg /inputfolder/grasp_params.json -pp /inputfolder/pproc.json -o /inputfolder/ -fr $p1 -fc $p2 -nspkc $p3 -igpu $p4

python demo_reconstruction.py -p /inputfolder/grasp_params.json -d /inputfolder/loader.csv -o /inputfolder/ -sid sub-1

python slice_to_vol.py -p /inputfolder/pproc.json -d /inputfolder/sub-1/*/raw-rec/ -o /inputfolder/sub-1/
