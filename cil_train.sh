
#!/bin/bash

#export python path

# USEFUL COMMANDS FOR CARLA DATA COLLECTION FROM DOCS
# sh CarlaUE4.sh /Game/Maps/Town01 -windowed -world-port=2000  -benchmark -fps=10
# python collect.py --data-path /home/whizz/Desktop/coil-datasets/Carla100 --data-configuration-name coil_training_dataset
# docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.8.4 /bin/bash CarlaUE4.sh  < Your list of parameters >


export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/coiltraine
export COIL_DATASET_PATH="/home/whizz/Desktop/coil-datasets"
echo "COIL_DATASET_PATH used: $COIL_DATASET_PATH"

# SIZE OF DATASETS
# Carla100: 6.180416666666665  hours of data
# Baseline: 10.62433333333333  hours of data

# Inspect checkpoint state_dict
# cp = torch.load('100_ef.pth')
# cp_list = [ '%s, %s' %(i, cp['state_dict'][i].shape) for i in cp['state_dict'] ]
# f = open('early_fusion.txt', 'w')
# f.write( '\n'.join(cp_list) )
# f.close()

# TRAINING
python /home/whizz/Desktop/coiltraine/coiltraine.py \
    --gpus=0 \
    --single-process="train" \
    --folder="carla10_exp" \
    --exp="control_1waypoint_CoILTrain" \
    --number-of-workers=12



# VALIDATION
# python /home/whizz/Desktop/coiltraine/coiltraine.py \
#     --gpus=0 \
#     --single-process="validation" \
#     --folder="carla10_exp" \
#     --exp="control_1waypoint" \
#     --val-datasets="CoILValCombined"



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

# python /home/whizz/Desktop/coiltraine/coiltraine.py \
#     --gpus=0 \
#     --single-process="drive" \
#     --folder="paper_carla100" \
#     --exp="carla100_1sttest" \
#     --drive-envs="NocrashNewWeatherTown_Town01" \
#     --docker='carlagear'



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

# paper_carla100_carla100_1sttest_700000_drive_control_output_CorlNewTown_Town02
# paper_carla100_carla100_1sttest_700000_drive_control_output_CorlNewWeatherTown_Town02
# paper_carla100_carla100_1sttest_700000_drive_control_output_NocrashNewWeatherTown_Town02

# METRIC_PATH="/home/whizz/Desktop/coiltraine/_benchmarks_results/"
# python /home/whizz/Desktop/coiltraine/carla08/driving_benchmark/results_printer.py \
#     --path="${METRIC_PATH}replicating_paper_results/paper_carla100_carla100_3rdtest_700000_drive_control_output_CorlNewWeatherTown_Town02" \
#     --exp_set_name="CorlNewWeatherTown"