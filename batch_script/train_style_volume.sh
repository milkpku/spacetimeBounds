machine=$1

train.py args/train_volume_up_walk.json --id walk-$machine-volume_up
train.py args/train_volume_up_cartwheel.json --id cartwheel-$machine-volume_up
train.py args/train_volume_down_walk.json --id walk-$machine-volume_down
train.py args/train_volume_down_cartwheel.json --id cartwheel-$machine-volume_down
