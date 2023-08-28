# PyGRASP: A library to reconstruct Dynamic Contrast enhanced MR Images acquired with radial sampling

PyGRASP is a Python library for temporally regularized iterative compressed sensing reconstruction (GRASP) for dynamic contrast enhanced MRI (DCE-MRI), developed by the team at Quantitative Intelligent Imaging Lab [QUIN](https://research.childrenshospital.org/quin/) at the radiology department of Boston Children's Hospital and Harvard Medical School. It assumes the k-space data is acquired by a Non-Cartesian, radial k-space sampling using the golden angle stack of stars trajectory.  The output is a dynamic set of volumes reconstructed at the desired temporal resolution. PyGRASP uses torchkbnufft library for a fast implementation of non-uniform Fast Fourier Transform (NUFFT) with Kaiser-Bessel gridding in PyTorch. PyGRASP uses GPU or CPU for parallelization of the NUFFT computations over the large number of dynamic volumes to be reconstructed over time.

There are several parameters that can be modified as needed including the number of radial lines (spokes) per volume, which determines the temporal resolution and also the number of volumes generated and the regularization parameter lambda of the GRASP reconstruction that determines the amount of temporal regularization. 

This library also provides the options to perform coil compression and/or coil removal before performing the reconstruction in order to speed up the reconstruction and also to improve image quality by removing coils that include severe artifacts.

PyGRASP is implemented completely in Python, facilitating flexible deployment with readable code and no complation. We also provide containerized version of the library in Dockers for GPU and CPU options for quick deployment.

If you use this library or its containerized Docker versions please cite our [ISMRM](https://submissions.mirasmart.com/ISMRM2023/Itinerary/PresentationDetail.aspx?evdid=2579) abstract (PyGRASP: A standalone python image reconstruction library for DCE-MRI acquired with radial sampling
Ariyurek C, Kocanaogullari A, Sari CT, Vasylechko S, Afacan O and Kurugol S. Proceedings of ISMRM 2023, p. 2404).

## Instructions to run dce_mri
There are three main parts - and Python scripts - of the reconstruction pipeline:
<ol>
  <li>convert_raw_to_dataset.py</li>
  Reads raw data file (currently only reads Siemens raw data), calculates coil sensitivity maps, writes k-space information and reconstruction parameters to the configuration files.
  <li>demo_reconstruction.py</li>
  Slice-by-slice GRASP reconstruction, reconstructed each slice is written to the file.
  <li>slice_to_vol.py</li>
  Reads reconstructed slices and generates the 4D volume. 
</ol>

## Running convert_raw_to_dataset.py

Parameters:
<ul>
  <li>-i, --path_csv: Path to mru-csv file which includes the directory of the raw data and patient info.</li>
  <li>-pg, --path_grasp_json: Path to grasp_params.json file which includes the parameters for the GRASP algorithm.</li>
  <li>-pp, --path_pproc_json: Path to pproc.json file which includes postprocessing parameters.</li>
  <li>-o, --path_dataset: Path to the dataset location (Output path directory to write k-space data and coil profiles).</li>
  <li>-igpu, --is_gpu: True to set NUFFT to gpu, False to run the method on CPU.</li>
  <li>-fr, --flag_reject_coil: True to reject outlier coils.</li>
  <li>-fc, --flag_compress_coil: True to compress coils with PCA.</li>
  <li>-nspkc, --num_spoke_coil: Number of spokes to estimate coil profiles.</li>
</ul>

Output of the script (Included in the output folder):
<ul>
  <li>info.txt: Mapping of subject name to ID.</li>
  <li>report.txt: Shows if read raw data and write k-space data are successful for the subject.</li>
  <li>loader.csv: Includes the path directories for k-space data and coil profiles.</li>
  <li>grasp_params.json: Overwritten based on the information from the raw data.</li>
  <li>pproc.json: Overwritten based on the information from the raw data.</li>
  <li>sub_X: Folder which includes slice-by-slice k-space data and coil profiles for subject “X”.</li>
</ul>

## Running demo_reconstruction.py

Parameters:
<ul>
  <li>-p, --root_json: Path to grasp_params.json file which includes the parameters for the GRASP algorithm. Overwritten based on the information from the raw data by convert_raw_to_dataset.py script.</li>
  <li>-d, --root_csv: Includes the path directories for k-space data and coil profiles. Created by convert_raw_to_dataset.py script.</li>
  <li>-o, --save_root: Output path directory to write reconstructed slices.</li>
  <li>-sid, --subject_id: Subject id (check loader.csv 1st column).</li>
</ul>

Output of the script (Included in the output folder):
<ul>
  <li>Slice-by-slice GRASP reconstructions are generated as .mat files in the output folder.</li>
</ul>

## Running slice_to_vol.py

Parameters:
<ul>
  <li>-p, --root_json: Path to pproc.json file which includes postprocessing parameters. Overwritten based on the information from the raw data by convert_raw_to_dataset.py.</li>
  <li>-d, --root_rec: Path that that includes raw_rec (Path directory to reconstructed slices .mat files).</li>
  <li>-o, --root_out: Output path directory.</li>
</ul>

Output of the script (Included in the output folder):
<ul>
  <li>Reconstructed 4D volume written as a nifti file.</li>
  <li>Grasp parameters used for generating the 4D volume.</li>
</ul>

Note that spv and lam values should be entered to grasp_params.json file by the user. Directory of MRI raw data file should be entered to the .csv file. Finally, flags for coil compression (fc) and coil rejection (fr) are inputs from the user from the command line to run convert_raw_to_dataset.py.

## Options for parallel processing
For each slice, the GRASP algorithm performs multiple iterations of iterative optimization for minimizing the loss function which includes a data consistency term and a regularization term for temporal regularization using the conjugate gradient descent algorithm to reduce the effect of undersamplign artifacts in the form of streaking. At each iteration a nonuniform Fourtier transform (NUFFT) is computed. To speed up the compuations, it is posisble to parallelize this step either using the GPU or the CPU. If the algorithms cannot locate available GPU resources, it will automatically run the processing using the available CPU resources.

## Configuration to run dce_mri on CPU and GPU on a Docker container (on centos7)

In order to start running dce_mri on a Docker container, your current directory should be the root directory of the repository in your machine.

## DEMO

You may run the code by cloning the repository, installing the environment, and running the scripts. An example script "pyGRASP_demo.sh" has been provided to run a demo case along with the raw data. Environment, code and input-output paths have to be edited in pyGRASP_demo.sh. Also, raw data address has to be updated in MRU.csv. Then, one can run the demo via "bash pyGRASP_demo.sh" command on terminal. There are two available data (DCE_MRI_MRU.dat) for DCE-MRI and they can be be downloaded from [figshare_data_1](https://figshare.com/articles/dataset/DCE_MRI_MRU_dat/20465637) and [figshare_data_2](https://figshare.com/articles/dataset/DCE-MRI_raw_data/22043195). 

Note that you would approximately need ~64 GB and ~16 GB of free memory are required for the first and the second dataset, respectively.

Steps to run pyGRASP demo using conda environment:
    
1) Download the data [figshare_data_1](https://figshare.com/articles/dataset/DCE_MRI_MRU_dat/20465637) or [figshare_data_2](https://figshare.com/articles/dataset/DCE-MRI_raw_data/22043195). The latter one is smaller in size due to having less number of channel measurements. 
    
2) Download/clone the repository pyGRASP
    
3) Create the environment from environment.yml (e.g., "conda env create -f environment.yml" in command line)
    
4) Dedicate a folder for inputOutput_path
    
5) Copy MRU.csv to inputOutput_path folder and edit it by replacing the directory path of the demo data 
    
6) Edit the environment, code, input/output paths in pyGRASP_demo.sh 
    
7) Run the pyGRASP_demo.sh (e.g., "bash pyGRASP_demo.sh" in command line)
    
8) You may find the reconstructed 4D DCE-MRI result as a NIfTI file in "inputOutput_path/sub-1/*date_time*-rec4D.nii.gz" along with the JSON file which records the parameters used in the reconstruction.  

Alternatively, steps to run pyGRASP demo using pip environment:
    
1) Download the data [figshare_data_1](https://figshare.com/articles/dataset/DCE_MRI_MRU_dat/20465637) or [figshare_data_2](https://figshare.com/articles/dataset/DCE-MRI_raw_data/22043195). The latter one is smaller in size due to having less number of channel measurements. 
    
2) Download/clone the repository pyGRASP
    
3) Create the environment from pip_requirements.txt and activate: 
    ### create virtual environment
    python3 -m venv $PWD/venv2

    ### activate
    source venv2/bin/activate

    ### install dependencies
    pip install -r pip_requirements.txt
    
4) Dedicate a folder for inputOutput_path
    
5) Copy MRU.csv to inputOutput_path folder and edit it by replacing the directory path of the demo data 
    
6) Edit inputOutput_path in pip_demo.sh
    
7) Go to pyGRASP-main folder and run pip_demo.sh (e.g., "bash pip_demo.sh" in command line)
    
8) You may find the reconstructed 4D DCE-MRI result as a NIfTI file in "inputOutput_path/sub-1/*date_time*-rec4D.nii.gz" along with the JSON file which records the parameters used in the reconstruction.  


Alternative to command prompt you can the code in a python editor/compiler. 

## CPU 

###  Step 1: Map folder containing configuration files

#### grasp_params.json
Json file containing grasp parameters

#### pproc.json
Json file containing postprocessing parameters

#### subject.csv
Csv file including the path of the raw data and patient info. The folder of the path for the raw data should be 'datafolder' since the absolute path of the raw data will be mapped as 'datafolder' in the Docker container (e.g., /datafolder/subject1.dat).
```
input_path=/path/to/folder/containing/grasp_params.json/subject.csv/pproc.csv/
```

###  Step 2: Map folder containing raw data
This path should be the folder containing the raw data given in subject.csv in the host machine. This path will be mapped as 'datafolder' in the Docker container and this path will be reachable in the Docker container as 'datafolder'. 
```
data_path=/path/to/folder/containing/raw/data/
```

###  Step 3: Build Docker image
'dockerfile_cpu' contains instructions to build the Docker image.
Build process copies the following files into the Docker container: 1) environment.yml (file containing required library signatures to run dce_mri), 2) dce_mri Python scripts, 3) run_grasp.sh (bash script containing Python calls to run dce_mri) .
```
docker build --no-cache -t dce_mri_cpu:latest -f dockerfile_cpu .
```
###  Step 4: Run Docker image
There are three parameters to run dce_mri using the Docker image: 1) #fr#: [flag_reject_coil] if true rejects outlier coils (default: False). 2) #fc#: [flag_compress_coil] if true compresses coils with PCA (default: False), 3) [num_spoke_coil] number of spokes to estimate coil profiles (default: 1300).
```
docker run -it --rm -v $input_path:/inputfolder -v $data_path:/datafolder dce_mri_cpu:latest bash run_grasp.sh #fr# #fc# #nspkc# #is_gpu#
```
Example:
```
docker run -it --rm -v $input_path:/inputfolder -v $data_path:/datafolder dce_mri_cpu:latest bash run_grasp.sh True True 500 False
```
The computation time was ~ 7.3 mins/slice, on CPU with 32 cores and 252GiB. 

## GPU

###  Step 1: Install NVIDIA Container Toolkit on your OS 
This integrates into Docker Engine to automatically configure your containers for GPU support. The CUDA version included in the first line of the dockerfile should be consistent with the version installed in your operating system.

#### update yum repos
```
# get repo list 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo
  
# for pre-releases, you need to enable the experimental repos of all dependencies:
sudo yum-config-manager --enable libnvidia-container-experimental
sudo yum-config-manager --enable nvidia-container-runtime-experimental

# to later disable the experimental repos of all dependencies, you can run:
sudo yum-config-manager --disable libnvidia-container-experimental
sudo yum-config-manager --disable nvidia-container-runtime-experimental
```

#### install nvidia container toolkit 
`yum install -y nvidia-docker2`

#### restart Docker
`sudo systemctl restart docker `

###  Step 2: Map folder containing configuration files

#### grasp_params.json
Json file containing grasp parameters

#### pproc.json
Json file containing postprocessing parameters

#### subject.csv
Csv file including the path of the raw data and patient info. The folder of the path for the raw data should be 'datafolder' since the absolute path of the raw data will be mapped as 'datafolder' in the Docker container (e.g., /datafolder/subject1.dat).
```
input_path=/path/to/folder/containing/grasp_params.json/subject.csv/pproc.csv/
```

###  Step 3: Map folder containing raw data
This path should be the folder containing the raw data given in subject.csv in the host machine. This path will be mapped as 'datafolder' in the Docker container and this path will be reachable in the Docker container as 'datafolder'. 
```
data_path=/path/to/folder/containing/raw/data/
```
###  Step 4: Build Docker image
'dockerfile_gpu' contains instructions to build the Docker image.
Build process copies the following files into the Docker container: 1) environment.yml (file containing required library signatures to run dce_mri), 2) dce_mri Python scripts, 3) run_grasp.sh (bash script containing Python calls to run dce_mri) .
```
docker build --no-cache -t dce_mri_gpu:latest -f dockerfile_gpu .
```
###  Step 5: Run Docker image
There are three parameters to run dce_mri using the Docker image: 1) #fr#: [flag_reject_coil] if true rejects outlier coils (default: False). 2) #fc#: [flag_compress_coil] if true compresses coils with PCA (default: False), 3) [num_spoke_coil] number of spokes to estimate coil profiles (default: 1300).
```
docker run -it --gpus all --rm -v $input_path:/inputfolder -v $data_path:/datafolder dce_mri_gpu:latest bash run_grasp.sh #fr# #fc# #nspkc# #is_gpu#
```
Example:
```
docker run -it --gpus all --rm -v $input_path:/inputfolder -v $data_path:/datafolder dce_mri_gpu:latest bash run_grasp.sh True True 500 True
```

The computation time was ~3.5 mins/slice on GPU, using a batch size of 10 ( GPU’s memory was 24564MiB). 

## Acknowledgements

This work was supported partially by the National Institute of Diabetic and Digestive and Kidney Diseases (NIDDK), National Institute of Biomedical Imaging and Bioengineering (NIBIB) under award numbers R01DK125561, R21DK123569 and R21EB029627.
