<img src='https://user-images.githubusercontent.com/41140561/65418411-56a91580-de37-11e9-872d-a978e98364b1.png' align="right">

# Multispectral Domain Invariant Image for Retrieval-based Place Recognition
- [ICRA2020 Paper](./MDII_paper.pdf)
- [ICRA2020 Presentation](https://www.slideshare.net/SejongRCV/multispectral-domain-invariant-image-for-retrievalbased-place-recognition-234803884)

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
nvidia-docker run -it -v $PWD:/workspace -p {port}:8888 -e NVIDIA_VISIBLE_DEVICES=all handchan/mdii /bin/bash
```
> if you have any problem with downloading the docker image, check this repository : https://hub.docker.com/r/handchan/mdii/tags

### Dataset

- Download Dataset

```sh
cd MDII
curl http://multispectral.sejong.ac.kr/ICRA2020_MDII/ICRA_MDII.tar.gz -o ICRA_MDII.tar.gz
tar -xzvf ICRA_MDII.tar.gz
```

- we support the pre-processed dataset.If you want to check the original dataset, refer to the following papers.
  - [All-day Vision Dataset (CVPRW 2015)](https://sites.google.com/site/ykchoicv/multispectral_vprice)
  - [All-day Vision Dataset (TITS 2018)](https://ieeexplore.ieee.org/document/8293689)

### Train

- Running train.py 

```sh
python train.py --name MDII --model MDII_gan --dataroot ./ICRA_MDII --gpu_ids GPU_NUM  --no_dropout --no_AA_BB
```

### Convert
- Running feat_c.py , make .npz file

```sh
### Convert train img to MDII
python feat_c.py --epoch {Epoch} --gpu_ids {GPU_NUM} --name MDII \ 
  --dataroot ./ICRA_MDII --no_dropout --model MDII_gan --phase train --eval
### Convert test img to MDII
python feat_c.py --epoch {Epoch} --gpu_ids {GPU_NUM} --name MDII \
  --dataroot ./ICRA_MDII --no_dropout --model MDII_gan --phase test --eval
```

### Evaluation
- Using Matlab vlfeat code. run rank.py
  - Download [vleat](https://www.vlfeat.org/) (our version is vlfeat-0.9.21)
  - Replace {vlfeat dir}/apps/recognition/ to [recognition_MDII](./recognition/)
  - Place your convert result name as {vlfeat dir}/MDII
    ```sh
    cd {vlfeat dir}
    ln -s {result dir} MDII (ex. ../../result/images/ICRA_MDII/{checkpoint name}/{epoch}/)
    # {vlfeat dir}
    # ├── apps
    # │   └── recognition
    # ├── data
    # │   ├── MDII -> ../../result/images/ICRA_MDII/{checkpoint name}/{epoch}/
    # │   │   ├── test
    # │   │   │   ├── rgb
    # │   │   │   ├── thr
    # │   │   ├── train
    # │   │   │   ├── rgb
    # │   │   │   ├── thr
    # ├──
    # ...
    ```
   - Run the Matlab code {vlfeat dir}/apps/recognition/experiments.m
   - Run the python code rank.py {workspace/rank.py}
   ```sh
   python rank.py --cache_path ./{vlfeat dir}/data_MDII_0604_200epoch/ex-MDII-vlad-aug
   # You can see the detail in python rank.py --help
   ```
### Citation

```
@INPROCEEDINGS{ICRA2020,
  author = {Daechan Han*, YuJin Hwang*, Namil Kim, Yukyung Choi},
  title = {Multispectral Domain Invariant Image for Retrieval-based Place Recognition},
  booktitle = {International Conference on Robotics and Automation(ICRA)},
  year = {2020}
}
```
