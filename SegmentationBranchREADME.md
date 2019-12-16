# Segmentation Branch README

segmentation images need to precede with 'segmentation_'
yaml file add MODEL_CONFIGURATION: branches: segmentation_head: 0

change branch weight to add one more weight for segmentation loss

data collection:

python collect.py --data-path /home/whizz/Desktop/coil-datasets/Carla100 --data-configuration-name coil_training_dataset

