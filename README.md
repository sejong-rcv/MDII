<img src='https://user-images.githubusercontent.com/41140561/65418411-56a91580-de37-11e9-872d-a978e98364b1.png' align="right">

# Multispectral Domain Invariant Image for Retrieval-based Place Recognition

### Dataset
- Pixel aligned Dataset
  - [All-day Vision Dataset (CVPRW 2015)](https://sites.google.com/site/ykchoicv/multispectral_vprice)
  - [All-day Vision Dataset (ITIS 2018)](https://ieeexplore.ieee.org/document/8293689)
- Location aligned Dataset
  

### Architecture
<img src='https://user-images.githubusercontent.com/41140561/65418697-fb2b5780-de37-11e9-9bbd-71b0e2e84940.png'>

### Train

- Make dataset directory 
```
python train.py --no_AA_BB --aef relu --gpu_ids 0 --name checkpoint_name --display_port portnum --loss_type En+SF --dataroot Datasetroot --gamma_identity 0 --no_dropout --model icra_gan
```

### Test
- Running feat_c.py , make .npz file
```
python feat_c.py --aef relu --checkpoints_dir ./checkpoints  --epoch epoch --gpu_ids 0 --name checkpoint_name --dataroot Datasetroot --no_dropout --model icra_gan

```

### Citation
- [ICRA2020 Paper Download](https://github.com/sejong-rcv/MDII/blob/master/ICRA2020_MDII.pdf)
- [IPIU2020 Paper Download]()

```
@INPROCEEDINGS{ICRA2020,
  author = {Daechan Han*, YuJin Hwang*, Namil Kim, Yukyung Choi},
  title = {Multispectral Domain Invariant Image for Retrieval-based Place Recognition},
  booktitle = {International Conference on Robotics and Automation(ICRA)},
  year = {2020}
}
@INPROCEEDINGS{IPIU2020,
  author = {한대찬*, 황유진*, 김남일, 최유경},
  title = {Multispectral Domain Invariant Image for Retrieval-based Place Recognition},
  booktitle = {제32회 영상처리 및 이해에 관한 워크샵},
  year = {2020}
}
```
