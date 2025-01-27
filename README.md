This is the repo for hand traking task using SAM2.1
# Installation
This is the instruction of how to intall the enviroment using conda.
The code requires **python>=3.10**, as well as **torch>=2.5.1** and **torchvision>=0.20.1**.
Please follow the instruction [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies.
The rest of the dependencies can be installed using below:

```bash
git clone https://github.com/Chen4549/HandTracking.git && cd HandTracking
conda env create -f environment.yml
```
The environment is now ready!
# Download the weights to local
In this project, I used this model checkpoint, which is the large one. Here is the download link below.
[sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

# Run the script
Before running the script, adjust some variable as needed.
Modify lines in the script [Tracking.py](Tracking.py#137)
```bash

