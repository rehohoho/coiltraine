

export COIL_DATASET_PATH="/mnt/ssd2/coil_datasets"
echo "COIL_DATASET_PATH used: $COIL_DATASET_PATH"


# DRIVE

python /home/whizz/Desktop/coiltraine/coiltraine.py \
    --gpus=1 \
    --single-process="drive" \
    --folder="nocrash" \
    --exp="resnet34imnet10S2" \
    --drive-envs="NocrashNewWeatherTown_Town02" \
    --docker='carlagear' \
    --host="172.18.0.2" \
    --number-of-workers=24 \
    --verbose \
    --port=7000 \
    --network_name="carlasimulator"

