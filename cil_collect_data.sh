
#!/bin/bash

#export python path

# sh CarlaUE4.sh /Game/Maps/Town01 -windowed -world-port=2000  -benchmark -fps=10
# python collect.py --data-path /home/whizz/Desktop/coil-datasets/Carla100 --data-configuration-name coil_training_dataset
# docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.8.4
# docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.8.4 /bin/bash CarlaUE4.sh  < Your list of parameters >


export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/coiltraine
export COIL_DATASET_PATH="/home/whizz/Desktop/coil-datasets"
echo "COIL_DATASET_PATH used: $COIL_DATASET_PATH"


# Carla100: 6.180416666666665  hours of data (50episodes)
# Baseline: 10.62433333333333  hours of data

python /home/whizz/Desktop/data-collector/multi_gpu_collection.py \
    --number_collectors=4 \
    --number_episodes=226 \
    --carlas_per_gpu=4 \
    --start_episode=94 \
    --data-configuration-name='coil_training_dataset' \
    --data-path='/home/whizz/Desktop/coil-datasets/Carla100' \
    --container-name='carlagear' \
    --town_name='1'