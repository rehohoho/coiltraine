
#!/bin/bash

#export python path

# sh CarlaUE4.sh /Game/Maps/Town01 -windowed -world-port=2000  -benchmark -fps=10
# python collect.py --data-path /home/whizz/Desktop/coil-datasets/Carla100 --data-configuration-name coil_training_dataset
# docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.8.4
# docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.8.4 /bin/bash CarlaUE4.sh  < Your list of parameters >


export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/coiltraine
export COIL_DATASET_PATH="/home/whizz/Desktop/coil-datasets"
echo "COIL_DATASET_PATH used: $COIL_DATASET_PATH"


# Carla100: 6.180416666666665  hours of data
# Baseline: 10.62433333333333  hours of data


# TRAINING
python /home/whizz/Desktop/coiltraine/coiltraine.py \
    --gpus=0 \
    --single-process="train" \
    --folder="waypoints" \
    --exp="test" \
    --number-of-workers=12



# VALIDATION
# python /home/whizz/Desktop/coiltraine/coiltraine.py \
#     --gpus=0 \
#     --single-process="validation" \
#     --folder="nocrash" \
#     --exp="resnet34imnet10S1" \
#     --val-datasets="baseline_dataset"



# DRIVE
# python /home/whizz/Desktop/coiltraine/coiltraine.py \
#     --gpus=0 \
#     --single-process="drive" \
#     --folder="nocrash" \
#     --exp="resnet34imnet10S1" \
#     --drive-envs="NocrashNewWeatherTown_Town02" \
#     --docker='carlagear'

# python /home/whizz/Desktop/coiltraine/coiltraine.py \
#     --gpus=0 \
#     --single-process="drive" \
#     --folder="nocrash" \
#     --exp="resnet34imnet100" \
#     --drive-envs="NocrashNewWeatherTown_Town02" \
#     --docker='carlagear'

# python /home/whizz/Desktop/coiltraine/coiltraine.py \
#     --gpus=0 \
#     --single-process="drive" \
#     --folder="nocrash" \
#     --exp="resnet34imnet10S1" \
#     --drive-envs="TestT1_Town02" \
#     --docker='carlagear'

# DE=( CorlNewTown_Town02 NocrashNewWeatherTown_Town02 )

# for i in "${DE[@]}"
# do
#     python /home/whizz/Desktop/coiltraine/coiltraine.py \
#         --gpus=0 \
#         --single-process="drive" \
#         --folder="paper_carla100" \
#         --exp="carla100_2ndtest" \
#         --drive-envs="$i" \
#         --docker='carlagear'
# done



# VISUALISE PATH
# python /home/whizz/Desktop/coiltraine/coilutils/visualise_path.py \
#     --gpus=0 \
#     --folder="vis_test" \
#     --exp="vis_test" \
#     --number_of_workers=4 \
#     --preload_dataset_name='visualisation_test'



# PRINT METRICS
#nocrash_resnet34imnet10S1_660000_drive_control_output_CorlNewWeatherTown_Town02
#nocrash_resnet34imnet10S1_660000_drive_control_output_NocrashNewWeatherTown_Town02

# METRIC_PATH="/home/whizz/Desktop/coiltraine/_benchmarks_results/"
# python /home/whizz/Desktop/coiltraine/carla08/driving_benchmark/results_printer.py \
#     --path="${METRIC_PATH}replicating_paper_results/nocrash_resnet34imnet10S1_660000_drive_control_output_NocrashNewWeatherTown_Town02" \
#     --exp_set_name="NocrashNewWeatherTown"