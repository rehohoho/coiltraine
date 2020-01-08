# Segmentation Branch README


yaml config file add MODEL_CONFIGURATION: branches: segmentation_head: 0

change branch weight to add one more weight for segmentation loss

###  Data Collection:

Carla Gear:

```
sh CarlaUE4.sh /Game/Maps/Town01 -windowed -world-port=2000  -benchmark -fps=10
```

```
python collect.py --data-path /home/whizz/Desktop/coil-datasets/Carla100 --data-configuration-name coil_training_dataset'''
```

### Training

Without segmentation

```
python coiltraine.py --gpus 0 --single-process train -e resnet34imnet --folder carla100
```

With segmentation

```
python coiltraine.py --gpus 0 --single-process train -e resnet34imnet --folder carla100 --use-seg-output
```

### Validation

```
python coiltraine.py --gpus 0 --single-process validation -e resnet34imnet --folder carla100 --val-datasets CoILVal1 --use-seg-output
```

### Live Eval

```
sh CarlaUE4.sh /Game/Maps/Town01 -windowed -world-port=2000  -benchmark -fps=10
```

```
python coiltraine.py --gpus 0 --single-process drive -e resnet34imnet --folder carla100 --use-seg-output --drive-envs TestT1_Town01 --docker carlagear
```