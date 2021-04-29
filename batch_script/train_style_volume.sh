machine=$1

python train.py args/train_volume_up_walk.json --id walk-$machine-volume_up
python train.py args/train_volume_up_cartwheel.json --id cartwheel-$machine-volume_up
python train.py args/train_volume_down_walk.json --id walk-$machine-volume_down
python train.py args/train_volume_down_cartwheel.json --id cartwheel-$machine-volume_down
