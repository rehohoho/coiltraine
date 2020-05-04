

export COIL_DATASET_PATH="/home/whizz/Desktop/coil-datasets"
echo "COIL_DATASET_PATH used: $COIL_DATASET_PATH"


# TRAINING
# python /home/whizz/Desktop/coiltraine/coiltraine.py \
#     --gpus=0 \
#     --single-process="train" \
#     --folder="carla10_exp" \
#     --exp="control_1waypoint_Carla100iterative" \
#     --number-of-workers=12



# VALIDATION
# Carla100_Val1_CorrectWeather 2.2215 hours
# Carla100_Val2_CorrectWeather 1.80875 hours
python /home/whizz/Desktop/coiltraine/coiltraine.py \
    --gpus=0 \
    --single-process="validation" \
    --folder="nocrash" \
    --exp="resnet34imnet100" \
    --val-datasets="Carla100_Val1_noise"

# python /home/whizz/Desktop/coiltraine/coiltraine.py \
#     --gpus=0 \
#     --single-process="validation" \
#     --folder="carla10_exp" \
#     --exp="control_1waypoint_Carla100iterative" \
#     --val-datasets="Carla100_Val1_nonoise"



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
#     --folder="carla10_exp" \
#     --exp="control_1waypoint_Carla100" \
#     --drive-envs="CorlNewWeatherTown_Town02" \
#     --docker='carlagear' 

# python /home/whizz/Desktop/coiltraine/coiltraine.py \
#     --gpus=0 \
#     --single-process="drive" \
#     --folder="carla10_exp" \
#     --exp="control_1waypoint_Carla100" \
#     --drive-envs="NocrashNewWeatherTown_Town02" \
#     --docker='carlagear'


