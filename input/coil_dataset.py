import os
import glob
import traceback
import collections
import sys
import math
import copy
import json
import random
import numpy as np

import torch
import cv2

from torch.utils.data import Dataset

from . import splitter
from . import data_parser

# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf
from coilutils.general import sort_nicely

"""
Hardcoded to read:
    1) forwardSpeed (0.8.x) OR velocity_x, velocity_y, velocity_z (0.9.x)
    2) gameTimestamp OR elapsed_seconds
    3) directions

Modify self.available_measurements and g_conf.REMOVE to read more measurements
see parse_remove_configuration() for g_conf.REMOVE implementation
"""


def parse_remove_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'remove'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key

    return name, conf_dict


def get_episode_weather(episode):
    with open(os.path.join(episode, 'metadata.json')) as f:
        metadata = json.load(f)
    print(" WEATHER OF EPISODE ", metadata['weather'])
    return int(metadata['weather'])


class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None, preload_name=None, available_measurements=None):
        
        # Setting the root directory for this dataset
        self.root_dir = root_dir

        # We add to the preload name all the remove labels
        if g_conf.REMOVE is not None and g_conf.REMOVE is not "None":
            name, self._remove_params = parse_remove_configuration(g_conf.REMOVE)
            self.preload_name = preload_name + '_' + name
            self._check_remove_function = getattr(splitter, name)
        else:
            self._check_remove_function = lambda _, __: False
            self._remove_params = []
            self.preload_name = preload_name

        print("preload Name ", self.preload_name)

        if self.preload_name is not None and os.path.exists(
                os.path.join('_preloads', self.preload_name + '.npy')):
            print("Loading from NPY ")
            self.sensor_data_names, self.measurements = np.load(
                os.path.join('_preloads', self.preload_name + '.npy'))
            print(self.sensor_data_names)
        else:
            
            if available_measurements == None:
                self.available_measurements = ['steer', 'throttle', 'brake']
            else:
                self.available_measurements = available_measurements

            print("Preloading data from %s" %root_dir)
            self.sensor_data_names, self.measurements = self._pre_load_image_folders(root_dir)

        print("preload Name ", self.preload_name)

        self.transform = transform
        self.batch_read_number = 0

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.

        Args:
            index:

        Returns:

        """
        try:
            img = _read_img_at_idx(index)

            measurements = self.measurements[index].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()

            measurements['rgb'] = img

            self.batch_read_number += 1
        except AttributeError:
            print ("Blank IMAGE")

            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb'] = np.zeros(3, 88, 200)

        return measurements

    def is_measurement_partof_experiment(self, measurement_data):

        # If the measurement data is not removable is because it is part of this experiment dataa
        return not self._check_remove_function(measurement_data, self._remove_params)

    def _read_img_at_idx(self, index, segmentation=False):
        if segmentation:
            img_path = os.path.join(self.root_dir,
                                    self.sensor_data_names[index].split('/')[-2],
                                    self.sensor_data_names[index].split('/')[-1].replace('RGB', 'SemanticSeg'))
        else:
            img_path = os.path.join(self.root_dir,
                                    self.sensor_data_names[index].split('/')[-2],
                                    self.sensor_data_names[index].split('/')[-1])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            print('Invalid path %s passed to _read_img_at_idx' %img_path)

        # Apply the image transformation
        if self.transform is not None:
            boost = 1
            img = self.transform(self.batch_read_number * boost, img)
        else:
            img = img.transpose(2, 0, 1)

        img = img.astype(np.float)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        if not segmentation:
            img = img / 255.
        return img

    def _get_final_measurement(self, speed, measurement_data, angle,
                               directions, available_measurements_dict):
        """
        Function to load the measurement with a certain angle and augmented direction.
        Also, it will choose if the brake is gona be present or if acceleration -1,1 is the default.

        Returns
            The final measurement dict
        """

        # Checks for augmentation is done in augment_measurement
        measurement_augmented = self.augment_measurement(copy.copy(measurement_data), angle,
                                                        speed, available_measurements_dict)
        
        if 'gameTimestamp' in measurement_augmented:
            time_stamp = measurement_augmented['gameTimestamp']
        else:
            time_stamp = measurement_augmented['elapsed_seconds']

        final_measurement = {}
        
        # We go for every available measurement, previously tested
        # and update for the measurements vec that is used on the training.
        for measurement, name_in_dataset in available_measurements_dict.items():
            # This is mapping the name of measurement in the target dataset
            final_measurement.update({measurement:
                data_parser.get_item_from_full_key(measurement_augmented, name_in_dataset)
            })

        # Add now the measurements that actually need some kind of processing
        final_measurement.update({'speed_module': speed / g_conf.SPEED_FACTOR})
        final_measurement.update({'directions': directions})
        final_measurement.update({'game_time': time_stamp})

        return final_measurement

    def _pre_load_image_folders(self, path):
        """
        Pre load the image folders for each episode, keep in mind that we only take
        the measurements that we think that are interesting for now.

        Args
            the path for the dataset

        Returns
            sensor data names: it is a vector with n dimensions being one for each sensor modality
            for instance, rgb only dataset will have a single vector with all the image names.
            float_data: all the wanted float data is loaded inside a vector, that is a vector
            of dictionaries.

        """

        episodes_list = glob.glob(os.path.join(path, 'episode_*'))
        sort_nicely(episodes_list)

        # Do a check if the episodes list is empty
        if len(episodes_list) == 0:
            raise ValueError("There are no episodes on the training dataset folder %s" % path)

        sensor_data_names = []
        float_dicts = []

        number_of_hours_pre_loaded = 0

        if len(glob.glob(os.path.join(path, '**', 'CameraRGB_*'))) > 0:
            center_prefix, left_prefix, right_prefix = 'CameraRGB_', 'LeftAugmentationCameraRGB_', 'RightAugmentationCameraRGB_'
        else:
            center_prefix, left_prefix, right_prefix = 'CentralRGB_', 'LeftRGB_', 'RightRGB_'
        
        print(center_prefix, left_prefix, right_prefix)

        for episode in episodes_list:

            print('Episode ', episode)

            # Check in sample measurements json for desired metrics specified in init
            available_measurements_dict = data_parser.check_available_measurements_modular(episode, self.available_measurements)
            if len(available_measurements_dict) != len(self.available_measurements):
                print('Warning! Not all desired metrics found in measurements! %s not found!' 
                %( [i for i in self.available_measurements if i not in available_measurements_dict.keys()] ))

            if number_of_hours_pre_loaded > g_conf.NUMBER_OF_HOURS: # achieved number of desired hours
                break

            # Get all measurements.json from this episode
            measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
            sort_nicely(measurements_list)
            if len(measurements_list) == 0:
                print("EMPTY EPISODE")
                continue

            count_added_measurements = 0

            for measurement in measurements_list[:-3]:

                data_point_number = measurement.split('_')[-1].split('.')[0]

                with open(measurement) as f:
                    measurement_data = json.load(f)

                # Only important metrics extracted, according to self.available_measurements and g_conf.REMOVE
                # see parse_remove_configuration()
                speed = data_parser.get_speed(measurement_data)

                directions = measurement_data['directions']
                final_measurement = self._get_final_measurement(speed, measurement_data, 0,
                                                                directions,
                                                                available_measurements_dict)

                if self.is_measurement_partof_experiment(final_measurement):
                    float_dicts.append(final_measurement)
                    rgb = 'CentralRGB_' + data_point_number + '.png'
                    sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                    count_added_measurements += 1

                # We do measurements augmentation for the left side cameras
                final_measurement = self._get_final_measurement(speed, measurement_data, -30.0,
                                                                directions,
                                                                available_measurements_dict)

                if self.is_measurement_partof_experiment(final_measurement):
                    float_dicts.append(final_measurement)
                    rgb = 'LeftRGB_' + data_point_number + '.png'
                    sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                    count_added_measurements += 1

                # We do measurements augmentation for the right side cameras
                final_measurement = self._get_final_measurement(speed, measurement_data, 30.0,
                                                                directions,
                                                                available_measurements_dict)

                if self.is_measurement_partof_experiment(final_measurement):
                    float_dicts.append(final_measurement)
                    rgb = 'RightRGB_' + data_point_number + '.png'
                    sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                    count_added_measurements += 1

            # Check how many hours were actually added
            last_data_point_number = measurements_list[-4].split('_')[-1].split('.')[0]
            number_of_hours_pre_loaded += (float(count_added_measurements / 10.0) / 3600.0)
            print(" Loaded ", number_of_hours_pre_loaded, " hours of data")


        # Make the path to save the pre loaded datasets
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')
        # If there is a name we saved the preloaded data
        if self.preload_name is not None:
            np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts

    def augment_directions(self, directions):

        if directions == 2.0:
            if random.randint(0, 100) < 20:
                directions = random.choice([3.0, 4.0, 5.0])

        return directions

    def augment_steering(self, camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras steering
        Args:
            camera_angle: the angle of the camera
            steer: the central steering
            speed: the speed that the car is going

        Returns:
            the augmented steering

        """
        time_use = 1.0
        car_length = 6.0

        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = g_conf.AUGMENT_LATERAL_STEERINGS * (
            math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))

        # print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer

    # TODO remove hardcode for augmentation checks
    def augment_measurement(self, measurements, angle, speed, available_measurements_dict):
        """
            Augment the steering of a measurement dict
            steer: (-1.0, 1.0) proportion of max_steer (70 deg for default vehicle)
            yaw: (0, 2pi), important to note that carla measurements are (-180, 180), anti-clockwise on xy-plane

            Current assignment is by name_in_measurements file directly
            Less hacky solution is to create assign_using_splitted_key
        """

        # Augments steering to correct error from camera angle, eg. -30 angle -> increase steer
        # We convert the speed to KM/h for the steer augmentation calculations
        if 'steer' in available_measurements_dict and angle != 0:
            steer_name = available_measurements_dict['steer']
            new_steer = self.augment_steering(angle, measurements[steer_name],
                                          3.6 * speed)
            measurements[steer_name] = new_steer

        # Augments yaw to heading of camera, eg. -30 angle -> lower yaw
        if 'rotation_yaw' in available_measurements_dict:
            yaw_name = available_measurements_dict['rotation_yaw']
            new_yaw = data_parser.get_item_from_full_key(measurements, yaw_name)
            if angle != 0:
                new_yaw += angle
            new_yaw = new_yaw % 360 * math.pi/180
            measurements[yaw_name] = new_yaw

        return measurements

    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]

    def check_coherence(self, index):
        """
        Used when targets include multiple waypoints
        Find index of frame where episode is changed, return NUMBER_OF_WAYPOINTS if does not change

        Args:
            index:  index of self.measurements to be read
        
        Returns:
            i:      index of frame where episode is changed
        """
        
        # end_frame = index + g_conf.NUMBER_OF_WAYPOINTS - self.max_index

        episode = None
        for i in range(0, g_conf.NUMBER_OF_WAYPOINTS*3, 3):     # for 3 cameras
            curr_episode = self.sensor_data_names[index + i].split('/')[-2]
            
            if episode == None:         #track episode of first frame
                episode = curr_episode
            
            if curr_episode != episode: #return frame where episode is different
                return(i/3)
        
        return(i/3+1)

    """
        Methods to interact with the dataset attributes that are used for training.
    """

    def extract_targets(self, data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in g_conf.TARGETS:
            targets_vec.append(data[target_name])

        if len(targets_vec) > 1:
            targets_vec = torch.cat(targets_vec, 1)
        else:
            targets_vec = torch.unsqueeze(targets_vec[0], dim = -1).type(torch.FloatTensor)
        
        return targets_vec

    def extract_inputs(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INPUTS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)

    def extract_intentions(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INTENTIONS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)

    def extract_ignore_waypoint_mask(self, data):
        """
        Used when targets include multiple waypoints
        Ignore waypoints that are not in the same episode as first frame

        Args:
            data:   dict from pytorch data_loader (see _getitem_)
        Returns:
            mask:   2D tensor (g_conf.BATCH_SIZE, g_conf.TARGETS) containing 1 and 0
        """
        
        sample_valid_len = data['incoherent'].tolist()
        n_outputs = len(g_conf.TARGET_KEYS)
        n_targets = len(g_conf.TARGETS)
        
        mask = torch.ones( [g_conf.BATCH_SIZE, n_targets] )
        
        for batch in range(g_conf.BATCH_SIZE):
            valid_ind = sample_valid_len[batch] *n_outputs
            if valid_ind != n_targets:
                mask[batch][int(valid_ind):] = 0
        
        return(mask)


class CoILDatasetWithSeg(CoILDataset):

    def __init__(self, root_dir, transform=None, preload_name=None, available_measurements=None):
        super().__init__(root_dir, transform, preload_name,
                        available_measurements = None)
        
        self.segmentation_n_class = 13 # https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-semantic-segmentation
        self.max_index = len(self.measurements)

    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.
        
        g_conf.INPUTS = ['speed_module']
        g_conf.TARGETS = ['steer%d', 'throttle%d', 'brake%d']
        get seg_ground_truth for calculation of loss at segmentation head
        """
        # TODO get ignore_map to consider end of dataset
        # work around should dataset approaches the end
        if (index + g_conf.NUMBER_OF_WAYPOINTS > self.max_index):
            index = random.randint(0, self.max_index - g_conf.NUMBER_OF_WAYPOINTS)
        
        try:
            img = self._read_img_at_idx(index, segmentation=False)

            single_channel_seg_img = self._read_img_at_idx(index, segmentation=True)[2]
            seg_ground_truth = np.zeros((self.segmentation_n_class, 88, 200))
            for i in range(self.segmentation_n_class):
                seg_ground_truth[i] = single_channel_seg_img == i * 1 #boolean np array cast to int
            seg_ground_truth = torch.from_numpy(seg_ground_truth).type(torch.FloatTensor)
            
            measurements = self.measurements[index].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()

            measurements['rgb'] = img
            measurements['seg_ground_truth'] = seg_ground_truth
            
            for target_key in g_conf.TARGET_KEYS:
                del measurements[target_key]
            
            for waypoint in range(g_conf.NUMBER_OF_WAYPOINTS):      #get target values for waypoints
                
                raw = self.measurements[index + waypoint].copy()
                for k, v in raw.items():
                    v = torch.from_numpy(np.asarray([v, ]))
                    raw[k] = v.float()

                for target_key in g_conf.TARGET_KEYS:
                    measurements['%s%d' %(target_key, waypoint)] = raw['%s' %target_key]
            
            #check for end of dataset, or episode break
            measurements['incoherent'] = self.check_coherence(index)
            
            self.batch_read_number += 1
        except AttributeError:
            print ("Blank IMAGE")

            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb'] = np.zeros(3, 88, 200)
            measurements['seg_ground_truth'] = np.zeros(self.segmentation_n_class, 88, 200)

        return measurements

    def extract_seg_gt(self, data):
        """
        Extract segmentation ground truths
        """
        targets_vec = []
        targets_vec.append(data['seg_ground_truth'])
        return torch.cat(targets_vec, 1)


class CoILDatasetWithWaypoints(CoILDataset):

    def __init__(self, root_dir, transform=None, preload_name=None):
        super().__init__(root_dir, transform, preload_name)
        
        self.max_index = len(self.measurements)
        
    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.
        
        g_conf.INPUTS = ['speed_module']
        g_conf.TARGETS = ['steer%d', 'throttle%d', 'brake%d']

        """
        # TODO get ignore_map to consider end of dataset
        # work around should dataset approaches the end

        # index runs in the same order as the preload .npy file (right mid left)
        # number of cameras is hardcoded at _pre_load_image_folders

        if (index + g_conf.NUMBER_OF_WAYPOINTS*3 > self.max_index):
            index = random.randint(0, self.max_index - g_conf.NUMBER_OF_WAYPOINTS*3)

        try:
            
            img = self._read_img_at_idx(index, segmentation=False)  #load image from folder
            
            measurements = self.measurements[index].copy()          #get measurements from preloaded
            for k, v in measurements.items():                       #turn them into tensors
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            
            measurements['rgb'] = img                               #save rgb tensor into measurements
            for target_key in g_conf.TARGET_KEYS:
                del measurements[target_key]
            
            for waypoint in range(g_conf.NUMBER_OF_WAYPOINTS):      #get target values for waypoints
                raw = self.measurements[index + 3*waypoint].copy()
                for k, v in raw.items():
                    v = torch.from_numpy(np.asarray([v, ]))
                    raw[k] = v.float()
                
                for target_key in g_conf.TARGET_KEYS:
                    measurements['%s%d' %(target_key, waypoint)] = raw['%s' %target_key]
            
            #check for end of dataset, or episode break
            measurements['incoherent'] = self.check_coherence(index)
            
            self.batch_read_number += 1

        except AttributeError:
            print ("Blank IMAGE")

            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb'] = np.zeros(3, 88, 200)
            measurements['seg_ground_truth'] = np.zeros(self.segmentation_n_class, 88, 200)

        return measurements
   
        
class CoILDatasetWithPathing(CoILDatasetWithSeg):
    
    def __init__(self, root_dir, use_seg_input, transform=None, preload_name=None):
        
        self.available_measurements = ['location_x', 'location_y', 'rotation_yaw']
        
        super().__init__(root_dir, transform, preload_name, 
                        available_measurements = self.available_measurements)
        
        self.max_index = len(self.measurements)
        self.incoherent_frame = -1
        self.use_seg_input = use_seg_input
        self.segmentation_n_class = 13 # https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-semantic-segmentation
    
    
    def _distance_between_waypoints(self, index, 
                                    local_x, local_y, local_yaw, 
                                    stopping_dist):
        dist = 0
        tar_index = index
        episode_changed = False
        
        while dist < stopping_dist:
            
            tar_index += 3      # 3 due to 3 cameras, hardcoded at preload

            if tar_index >= self.max_index - 4:  # handle end of dataset
                episode_changed = True
                break
            else:
                tar = self.measurements[tar_index]
                tar_x = tar['location_x']
                tar_y = tar['location_y']

                dist_x = tar_x - local_x
                dist_y = tar_y - local_y
                dist = dist_x * dist_x + dist_y * dist_y

                if dist > 10000:            #handle change in episode
                    episode_changed = True

        # measurements['rotation_yaw'] modified from (-180,180) to (0,2pi) in augment_measurements
        # anti-clockwise on xy plane positive (UE4 default), 0 is north heading
        if episode_changed:
            angle_diff = 0
        else:
            tar_yaw = -math.atan(dist_x/dist_y)
            
            if dist_y < 0:                      # compute for a unit circle
                tar_yaw += math.pi
            
            tar_yaw %= (2 * math.pi)            # scale to (0, 2pi) similar to augment_measurements
            
            angle_diff = tar_yaw - local_yaw
            
            if abs(angle_diff) > 1.5*math.pi:   # handle crossing the north heading
                if angle_diff > 0:
                    angle_diff = angle_diff - 2*math.pi
                else:
                    angle_diff = angle_diff + 2*math.pi
        
        return angle_diff


    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.
        
        g_conf.INPUTS = ['speed_module']
        g_conf.TARGETS = ['steer%d', 'throttle%d', 'brake%d']
        """
        # TODO get ignore_map to consider end of dataset
        # work around should dataset approaches the end
        if (index + g_conf.NUMBER_OF_WAYPOINTS > self.max_index):
            index = random.randint(0, self.max_index - g_conf.NUMBER_OF_WAYPOINTS)

        try:
            
            img = self._read_img_at_idx(index, segmentation=False)  #load image from folder
            
            measurements = self.measurements[index].copy()          #get measurements from preloaded
            for k, v in measurements.items():                       #turn them into tensors
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()

            measurements['rgb'] = img                               #save rgb tensor into measurements
            
            if self.use_seg_input:
                single_channel_seg_img = self._read_img_at_idx(index, segmentation=True)[2]
                seg_ground_truth = np.zeros((self.segmentation_n_class, 88, 200))
                for i in range(self.segmentation_n_class):
                    seg_ground_truth[i] = single_channel_seg_img == i * 1 #boolean np array cast to int
                seg_ground_truth = torch.from_numpy(seg_ground_truth).type(torch.FloatTensor)
                measurements['seg_ground_truth'] = seg_ground_truth
            
            for i in range(len(g_conf.PATHING_SQDISTANCE)):
                local_x = measurements['location_x']
                local_y = measurements['location_y']
                angle_diff = self._distance_between_waypoints(index, local_x, local_y, measurements['rotation_yaw'],
                                                            stopping_dist = g_conf.PATHING_SQDISTANCE[i])
                measurements['angle%d' %i] = torch.from_numpy(np.asarray([angle_diff,])
                                                    ).type(torch.FloatTensor)

            #extract ignore waypoint mask cannot be used, since it observes future steps by time, not distance
            measurements['incoherent'] = g_conf.NUMBER_OF_WAYPOINTS
            
            self.batch_read_number += 1

        except AttributeError:
            print ("Blank IMAGE")

            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb'] = np.zeros(3, 88, 200)
            measurements['seg_ground_truth'] = np.zeros(self.segmentation_n_class, 88, 200)

        return measurements