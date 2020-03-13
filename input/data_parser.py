import glob
import os
import json
import numpy as np
"""
Module used to check attributes existent on data before incorporating them
to the coil dataset
"""


def orientation_vector(measurement_data):
    pitch = np.deg2rad(measurement_data['rotation_pitch'])
    yaw = np.deg2rad(measurement_data['rotation_yaw'])
    orientation = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
    return orientation


def forward_speed(measurement_data):
    vel_np = np.array([measurement_data['velocity_x'], measurement_data['velocity_y'],
                       measurement_data['velocity_z']])
    speed = np.dot(vel_np, orientation_vector(measurement_data))

    return speed


def get_speed(measurement_data):
    """ Extract the proper speed from the measurement data dict """

    # If the forward speed is not on the dataset it is because speed is zero.
    if 'playerMeasurements' in measurement_data and \
            'forwardSpeed' in measurement_data['playerMeasurements']:
        return measurement_data['playerMeasurements']['forwardSpeed']
    elif 'velocity_x' in measurement_data:  # We have a 0.9.X data here
        return forward_speed(measurement_data)
    else:  # There is no speed key, probably speed is zero.
        return 0


def find_full_keys_recursively(dictionary, parent = None):
    for key, value in dictionary.items():
        
        if parent is not None:
            full_key = '_'.join([parent, key])
        else:
            full_key = key
        
        if isinstance(value, dict):
            yield (full_key, value)
            yield from find_full_keys_recursively(value, parent = full_key)
        else:
            yield (full_key, value)


def get_item_from_full_key(dictionary, key):
    """ Get item from key if key exists
    Get item from splitted key otherwise
    Requirements: key names has to be joined by '_' (hardcoded)

    TODO: non-hacky solution is to do assign_by_full_key, current assignment uses name_in_dataloader
    """

    if key in dictionary.keys():
        return dictionary[key]

    full_key = key.split('_')
    final_item = dictionary

    for k in full_key:
        if k not in final_item.keys():
            print('Warning! %s is not found in measurements' %full_key)
            return 0
        final_item = final_item[k]
    
    return final_item


def check_available_measurements_modular(episode, keys_to_retrieve):
    """ Checks sample measurement for key
    Requirements: key names has to be joined by '_' (hardcoded)
    """
    
    measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
    
    with open(measurements_list[0]) as f:   # Open a sample measurement
        measurement_data = json.load(f)
    
    available_measurements = {}
    for key, item in find_full_keys_recursively(measurement_data):
        for tar_key in keys_to_retrieve:
            if tar_key == key[-len(tar_key):] :
                
                # shortest path to desired metric precedes (eg. brake VS hand_brake)
                if tar_key in available_measurements.keys() and \
                    len(available_measurements[tar_key]) < len(key):
                    continue

                available_measurements.update({tar_key: key})

    return available_measurements


# deprecated
def check_available_measurements(episode):
    """ Try to automatically check the measurements
        The ones named 'steer' are probably the steer for the vehicle
        This needs to be made more general to avoid possible mistakes on dataset reading
    """

    measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
    # Open a sample measurement
    with open(measurements_list[0]) as f:
        measurement_data = json.load(f)

    available_measurements = {}
    for meas_name in measurement_data.keys():

        # Add steer
        if 'steer' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'steer': meas_name})

        # Add Throttle
        if 'throttle' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'throttle': meas_name})

        # Add brake ( Not hand brake)
        if 'brake' in meas_name and 'noise' not in meas_name and 'hand' not in meas_name:
            available_measurements.update({'brake': meas_name})

        # add game time

    return available_measurements

