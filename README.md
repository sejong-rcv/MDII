<img src='https://user-images.githubusercontent.com/41140561/65418411-56a91580-de37-11e9-872d-a978e98364b1.png' align="right">

# Multispectral Domain Invariant Image for Retrieval-based Place Recognition

### Dataset
- Pixel aligned Dataset
  - [All-day Vision Dataset (CVPRW 2015)](https://sites.google.com/site/ykchoicv/multispectral_vprice)
  - [All-day Vision Dataset (ITIS 2018)](https://ieeexplore.ieee.org/document/8293689)
- Location aligned Dataset
  

### Architecture
<img src='https://user-images.githubusercontent.com/41140561/65418697-fb2b5780-de37-11e9-9bbd-71b0e2e84940.png'>

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## Getting Started


### Installation

- Clone Repo

```sh
git clone https://github.com/sejong-rcv/MDII
cd MDII
```
### Docker 

- Prerequisite 
  - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 
- Option
  - visdom port number
   
```sh

nvidia-docker run -it -v $PWD:/workspace \
    -e NVIDIA_VISIBLE_DEVICES=all \
    handchan/mdii /bin/bash
```
### Dataset

- Download Dataset

```sh
cd MDII
curl http://multispectral.sejong.ac.kr/ICRA2020_MDII/CVPRW_kaist_data.tar.gz -o CVPRW_kaist_data.tar.gz
tar -xzvf CVPRW_kaist_data.tar.gz
```

### Train

- Running train.py 

```sh
python train.py --no_AA_BB --aef relu --gpu_ids GPU_NUM --name MDII --display_port 8888 --loss_type En+SF --dataroot ./CVPRW_kaist --gamma_identity 0 --no_dropout --model MDII_gan
```

### Convert
- Running feat_c.py , make .npz file

```sh
### Convert train img to MDII
python feat_c.py --aef relu --epoch Epoch --gpu_ids GPU_NUM --name checkpoint_name --dataroot path/to/data/CVPRW_kaist --no_dropout --model MDII_gan --phase train
### Convert test img to MDII
python feat_c.py --aef relu --epoch Epoch --gpu_ids GPU_NUM --name checkpoint_name --dataroot path/to/dataCVPRW_kaist --no_dropout --model MDII_gan --phase test
```

### Citation
- [ICRA2020 Paper]

```
@INPROCEEDINGS{ICRA2020,
  author = {Daechan Han*, YuJin Hwang*, Namil Kim, Yukyung Choi},
  title = {Multispectral Domain Invariant Image for Retrieval-based Place Recognition},
  booktitle = {International Conference on Robotics and Automation(ICRA)},
  year = {2020}
}
```
