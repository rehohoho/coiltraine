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


def check_kinematic_measurements(episode):
    measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
    # Open a sample measurement
    with open(measurements_list[0]) as f:
        measurement_data = json.load(f)

    available_measurements = {}
    for meas_name in measurement_data.keys():
        
        # Add location
        if 'location_x' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'location_x': meas_name})
        if 'location_y' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'location_y': meas_name})
        
        # Add velocity
        if 'velocity_x' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'velocity_x': meas_name})
        if 'velocity_y' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'velocity_y': meas_name})
        
        # Add acceleratoin
        if 'acceleration_x' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'acceleration_x': meas_name})
        if 'acceleration_y' in meas_name and 'noise' not in meas_name:
            available_measurements.update({'acceleration_y': meas_name})
        
        # Add yaw
        if 'rotation_yaw' in meas_name and 'noise' not in meas_name and 'hand' not in meas_name:
            available_measurements.update({'yaw': meas_name})
        if 'rotation_pitch' in meas_name and 'noise' not in meas_name and 'hand' not in meas_name:
            available_measurements.update({'pitch': meas_name})

        # add game time

    return available_measurements


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

