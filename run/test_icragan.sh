set -ex
python feat.py --dataroot ./datasets/feature/AM05 --name check_1 --model icra_gan --phase test --no_dropout --gpu_ids 0 --nef 64 --nef 128
