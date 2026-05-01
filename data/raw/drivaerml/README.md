---
viewer: false
license: cc-by-sa-4.0
---

DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics
-------

Contact: 
----------
Neil Ashton (contact@caemldatasets.org)

Website:
----------
https://caemldatasets.org

Summary:
-------

Machine Learning (ML) has the potential to revolutionise the field of automotive aerodynamics, enabling split-second flow predictions early in the design process. 
However, the lack of open-source training data for realistic road cars, using high-fidelity CFD methods, represents a barrier to their development.
To address this, a high-fidelity open-source (CC-BY-SA) public dataset for automotive aerodynamics has been generated, based on 500 parametrically morphed variants of the widely-used DrivAer notchback generic vehicle. Mesh generation and scale-resolving CFD was executed using consistent and validated automatic workflows representative of the industrial state-of-the-art. Geometries and rich aerodynamic data are published in open-source formats. To our knowledge, this is the first large, public-domain dataset for complex automotive configurations generated using high-fidelity CFD. 

CFD Solver:
----------
All cases were run using the open-source finite-volume code OpenFOAM v2212 with custom modifications by UpstreamCFD. Please see the paper below for full details on the code and validation:

How to cite this dataset:
----------------
In order to cite the use of this dataset please cite the paper below which contains full details on the dataset.

''
@article{ashton2024drivaer,
    title = {{DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics}},
    year = {2024},
journal = {arxiv.org},
    url={https://arxiv.org/abs/2408.11969},
    author = {Ashton, N., Mockett, C., Fuchs, M., Fliessbach, L., Hetmann, H., Knacke, T., Schonwald, N.,
Skaperdas, V., Fotiadis, G., Walle, A., Hupertz, B., and Maddix, D}
}
''

Files:
-------
Each folder (e.g run1,run2...run"i" etc) corresponds to a different geometry that contains the following files where "i" is the run number: 
* geometry stl (~135mb): drivaer_i.stl 
* reference values for each geometry: geo_ref_i.csv 
* reference geometry for each geometry: geo_parameters_i.csv 
* Boundary VTU (~500mb): boundary_i.vtp 
* Volume field VTU (~50GB): volume_i.vtu ( please note on HuggingFace this is split into part 1 and part2 - please cat them together to create the volume_i.vtu)
* forces/moments time-averaged (using varying frontal area/wheelbase): force_mom_i.csv 
* forces/moments time-averaged (using constant frontal area/wheelbase): force_mom_constref_i.csv 
* slices: folder containing .vtp slices in x,y,z that contain flow-field variables 
* Images: This folder contains images of various flow variables (e.g. Cp, CpT, UMagNorm) for slices of the domain at X, Y, and Z locations (M signifies minus, P signifies positive), as well as on the surface. It also includes evaluation plots of the time-averaging of the force coefficients (via the tool MeanCalc) and a residual plot illustrating the convergence.

In addition to the files per run folder, there are also: 
* openfoam_meshes : this folder contains the OpenFOAM meshes (in OpenFOAM format) used for these simulations. The 0 and system folders are just the default output from ANSA and were not those used in this study. Please refer to the arxiv paper for full details of the CFD setup. We hope that by providing the meshes, groups may wish to expand the dataset as they see fit.  
* force_mom_all.csv : forces/moments time-averaged (using varying frontal area/wheelbase) for all runs 
* force_mom_constref_all.csv : forces/moments time-averaged (using constant frontal area/wheelbase) for all runs 
* geo_parameters_all.csv: reference geometry values for each geometry for all runs 

How to download:
----------------

The dataset is now available on HuggingFace. Below are some examples of how to download all or selected parts of the dataset. Please refer to the HuggingFace documentation for other ways to accessing the dataset and building workflows.

Example 1: Download all files (~31TB)
----------
Please note you’ll need to have git lfs installed first, then you can run the following command:

git clone git@hf.co:datasets/neashton/drivaerml

Example 2: only download select files (STL,images & force and moments):
--------

Create the following bash script that could be adapted to loop through only select runs or to change to download different files e.g boundary/volume.

```
#!/bin/bash

# Set the path and prefix
HF_OWNER="neashton"
HF_PREFIX="drivaerml"

# Set the local directory to download the files
LOCAL_DIR="./drivaer_data"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Loop through the run folders from 1 to 500
for i in $(seq 1 500); do
    RUN_DIR="run_$i"
    RUN_LOCAL_DIR="$LOCAL_DIR/$RUN_DIR"

    # Create the run directory if it doesn't exist
    mkdir -p "$RUN_LOCAL_DIR"

    # Download the drivaer_i.stl file
    wget "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/drivaer_$i.stl" -O "$RUN_LOCAL_DIR/drivaer_$i.stl"

    # Download the force_mom_i.csv file
    wget "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/force_mom_$i.csv" -O "$RUN_LOCAL_DIR/force_mom_$i.csv"

done
```

Credits
-----

* CFD solver and workflow development by Charles Mockett, Marian Fuchs, Louis Fliessbach, Henrik Hetmann, Thilo Knacke & Norbert Schonwald (UpstreamCFD)
* Geometry parameterization by Vangelis Skaperdas, Grigoris Fotiadis (BETA-CAE Systems) & Astrid Walle (Siemens Energy) 
* Meshing development workflow by Vangelis Skaperdas & Grigoris Fotiadis (BETA-CAE Systems) 
* DrivAer advise and consultation by Burkhard Hupertz (Ford) 
* Guidance on dataset preparation for ML by Danielle Maddix (Amazon Web Services - now NVIDIA)
* Simulation runs, HPC setup and dataset preparation by Neil Ashton (Amazon Web Services - now NVIDIA) 

License
----
This dataset is provided under the CC BY SA 4.0 license, please see LICENSE.txt for full license text.

version history:
---------------
* 04/03/2025 - Now available on HuggingFace!
  
* 11/11/2024 - the 15 of the 17 cases that were missing are being considered for use as a blind study. For the time-being these are available but password protected in the file blind_15additional_cases_passwd_required.zip. Once we setup a benchmarking sysystem we will provide details on how people can test their methods against these 15 blind cases. 

* 08/10/2024 - The OpenFOAM meshes (in OpenFOAM format) that were generated in ANSA have been uploaded to the openfoam_meshes folder. The 0 and system folders are just the default output from ANSA and were not those used in this study. Please refer to the arxiv paper for full details of the CFD setup. We hope that by providing the meshes, groups may wish to expand the dataset as they see fit.

* 10/09/2024 - Run_0 has been added as a blind study for the AutoCFD4 workshop. Post-workshop the results from this additional run will be uploaded to the dataset.

* 29/07/2024 - Note: please be aware currently runs 167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403, 473 are not in the dataset. 

* 03/05/2024 - draft version produced 

