# Repository for Developing Temporal Action Localization for the AI City 2023 Naturalistic Driving Action Recognition Challenge
More information about this particular challenge can be found [(here)](https://www.aicitychallenge.org/2023-challenge-tracks/)

For an in-depth explanation of the system architecture, please refer to the paper inside this repo (paper.pdf)

## Requirements
This repo uses [PySlowFast](https://github.com/facebookresearch/SlowFast) and [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2) as the codebase. 

Install the following: 
- Python >= 3.8.10, pytorch == 1.13.1+c117, pandas, tqdm, scikit-learn, decord, tensorrt (may have to upgrade pip3 first), torchmetrics
- FFmpeg >= 4.2.7 
- GNU Parallel 
- Follow the INSTALL.md inside the first slowfast folder for PySlowFast reqs; do NOT clone or build PySlowFast
- Look at requirements.txt for VideoMAEv2 and install necessary libs

For video and decoder functionality to work, torchvision MUST be compiled & built from source (used v0.14.1+c117): 
- uninstall FFmpeg if you already have it, then reinstall it with the following command:
```
sudo apt install ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev libswresample-dev libpostproc-dev libjpeg-dev libpng-dev
```
- clone the torchvision release compatible with your pytorch version
- add this line to top of setup.py: 
```python
sys.path.append("/home/vislab-001/.local/lib/python3.8/site-packages")
```
- to make sure the setup.py has full permissions, use the following command:
```
sudo chmod 777 {path to torchvision repo}
```
- run the setup.py, if there are more permission errors, simply chmod 777 the folder location indicated by the errors


## Setup
We use 2 NVIDIA A5000 RTX GPUs with 24GB VRAM each, and 32GB RAM.

- Download the data for track #3 [here](https://www.aicitychallenge.org/2023-data-and-evaluation/)
- Download the VideoMAEv2 K710 checkpoint [(link)](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/distill/vit_b_k710_dl_from_giant.pth) and place file in checkpoints folder (must edit the checkpoint path in slowfast config)
- create an empty folder within the repo where the video clips will be dumped 

Note:
In [Envy_AI_City/VideoMAEv2/dataset/datasets.py](https://github.com/CarrotPeeler/Envy_AI_City/blob/main/VideoMAEv2/dataset/datasets.py),
you must change the *sys.path.insert* statement at the top to match your file system.

## Data Preparation
- In the prepocessing folder, there are two scripts to extract clips from both the A1 (train) and A2 (test) datasets. Please edit the following parameters as well as others in both scripts. 
```python
videos_loadpath = "/path_to_A1/SET-A1"
clips_savepath = "/path_to_data/data_dir"
```
- run the following for A1 Train (optionally run inside tmux):
```
python3 preprocessing/prepare_train_data_multiview.py < /dev/null > ffmpeg_log.txt 2>&1 &
```
- run the following for A2 Inference (optionally run inside tmux):
```
python3 preprocessing/prepare_inf_data_multiview.py < /dev/null > ffmpeg_log.txt 2>&1 &
```

## Training
- edit the config in VideoMAEv2/configs/vit_b_k710_multiview.sh
    - adjust NUM_GPUS, NUM_WORKERS, etc. based on your PC specs
- cd into VideoMAEv2 folder
- train VideoMAEv2 (optionally run inside tmux):
```
bash configs/vit_b_k710_multiview_5fold.sh < /dev/null > train_log.txt 2>&1 &
```

## Model ZOO
TBD...

## Inference
For localization inference, we obtain prediction probabilities using 64 frame action proposals, generated every 16 frames of a given video. To understand how we perform inference, please look at [Envy_AI_City/slowfast/tools/run_tal.py](https://github.com/CarrotPeeler/Envy_AI_City/blob/main/slowfast/tools/run_tal.py).

- edit the config in slowfast/slowfast/configs (TAL_inf.yaml)
    - make sure checkpoint directories point to correct file path    
    - make sure correct model checkpoints (.pt files) are in slowfast/checkpoints with correct folder structure below
```
├── slowfast
│   ├── checkpoints
│   │   ├── dash
|   │   │   ├── fold_0
|   |   │   │   ├── dash_fold_0.pt
|   |   │   │   ├── dash_fold_1.pt
|   |   │   │   ├── ... 
|   │   │   ├── fold_1
|   │   │   ├── ... 
│   │   ├── rear
|   │   │   ├── fold_0
|   │   │   ├── fold_1
|   │   │   ├── ... 
│   │   ├── right
|   │   │   ├── fold_0
|   │   │   ├── fold_1
|   │   │   ├── ... 
```  

- cd into outermost slowfast folder
- perform inference:
```
python3 tools/run_net.py --cfg configs/TAL_inf.yaml DATA.PATH_TO_DATA_DIR .
```
- Once inference is complete, probabilities will be saved to a folder => slowfast/inference/probs.

## Post-Processing 
To process prediction probabilities for Temporal Action Localization, first edit the arguments in [Envy_AI_City/slowfast/inference/post_process.py](https://github.com/CarrotPeeler/Envy_AI_City/blob/main/slowfast/inference/post_process.py).

Run the following:
```
cd slowfast (if you haven't already)
python3 inference/post_process.py
```
The final submission results for the evaluation server can be found in inference/submission_files/submission_file.txt

## Results
After submitting TAL results to the track #3 evaluation server for the AI City challenge [(link)](https://www.aicitychallenge.org/2023-evaluation-system/), the above methods net a final score of **0.6010**. Although this project was completed just after the 2023 challenge ended, this score ranks 6th overall on the public leaderboards for the A2 dataset. 

## Contact
For any questions, send an email to: jared24mc@gmail.com