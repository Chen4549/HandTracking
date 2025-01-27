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
Modify lines in the script [Tracking.py](Tracking.py#L137-L145)
```bash
    sam2_checkpoint = '/home/hc4549/HandTracking/sam2.1_hiera_large.pt' #Weight path (downloaded from official SAM web)
    model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml' #Config path

    video_dir = 'videoframe' #Directory that contian all the frames
    #"Clicks" coordination, should be in this form, 1 means positive click, while 0 means negative click.
    point_dict = {1 : [[580, 200],[800,400],[200,600],[900,700]], 
                  0 : [[630,100]]}

    output_dir = 'segmented_frames' #Output folder
```
Above is the variable need to be adjusted:  
sam2_checkpoint is the path to the weight.  
mdoel_cfg is the model config which is under the folder 'configs/sam2.1'.  
video_dir is the directory that contains all the frames.  
point_dict is the coordinate of the 'clicks' for both positive and negative clicks.  
output_dir is the desired output path, where it will save the segmented frame for the video.
