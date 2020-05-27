set -ex
python feat_c.py --aef relu --epoch Epoch --gpu_ids 3 --name e907_3 --dataroot ICRA_GTA_final/feature/AM09 --no_dropout --model icra_gan --phase train
python feat_c.py --aef relu --epoch Epoch --gpu_ids 3 --name e907_3 --dataroot ICRA_GTA_final/feature/AM05 --no_dropout --model icra_gan --phase test
