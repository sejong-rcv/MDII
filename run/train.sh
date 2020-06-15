set -ex
python train.py --no_AA_BB --aef relu --gpu_ids 0 --name MDII --display_port 8888 --loss_type En+SF --dataroot ./CVPRW_kaist --gamma_identity 0 --no_dropout --model MDII_gan