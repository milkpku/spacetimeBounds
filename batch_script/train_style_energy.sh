machine=$1

train.py args/train_energy_down_run.json --id run-$machine-energy_down
train.py args/train_energy_down_dance.json --id dance-$machine-energy_down
train.py args/train_energy_down_cartwheel.json --id cartwheel-$machine-energy_down
train.py args/train_energy_up_run.json --id run-$machine-energy_up
train.py args/train_energy_up_dance.json --id dance-$machine-energy_up
train.py args/train_energy_up_cartwheel.json --id cartwheel-$machine-energy_up
