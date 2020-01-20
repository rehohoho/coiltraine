

export PYTHONPATH=$PYTHONPATH:/home/whizz/Desktop/coiltraine

# FIND THE USB PORT (/dev/asdf)
# for sysdevpath in $(find /sys/bus/usb/devices/usb*/ -name dev); do
#     (
#         syspath="${sysdevpath%/dev}"
#         devname="$(udevadm info -q name -p $syspath)"
#         [[ "$devname" == "bus/"* ]] && continue
#         eval "$(udevadm info -q property --export -p $syspath)"
#         [[ -z "$ID_SERIAL" ]] && continue
#         echo "/dev/$devname - $ID_SERIAL"
#     )
# done

# INFERENCE
python /home/whizz/Desktop/coiltraine/inference.py \
    --output_folder="/home/whizz/Desktop/coiltraine" \
    --checkpoint_file="/home/whizz/Desktop/coiltraine/_logs/nocrash/resnet34imnet100/checkpoints/700000.pth" \
    --config_file="/home/whizz/Desktop/coiltraine/configs/nocrash/resnet34imnet100.yaml"