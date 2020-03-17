import os
import sys
import random
import time
import traceback
import numpy as np
import torch
import torch.optim as optim

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss, adjust_learning_rate_auto
from input import CoILDatasetWithSeg, CoILDatasetWithWaypoints, CoILDatasetWithPathing, \
                  Augmenter, select_balancing_strategy
from logger import coil_logger
from coilutils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, \
                                    check_loss_validation_stopped


carla_cm = np.array([
    (  0,  0,  0),  #none
    ( 70, 70, 70), #buildings
    (190,153,153), #fences
    (250,170,160), #other
    (220, 20, 60), #pedestrains
    (153,153,153), #poles
    (157,234, 50), #road lines
    (128, 64,128), #roads
    (244, 35,232), #sidewalks
    (107,142, 35), #vegetation
    (  0,  0,142), #vehicles/car
    (102,102,156), #walls
    (220,220,  0)  #traffic signs
], dtype = np.float32)


def _get_targets():

    # Change number of targets according to number of waypoints
    g_conf.TARGET_KEYS = g_conf.TARGETS
    targets = g_conf.TARGETS * g_conf.NUMBER_OF_WAYPOINTS

    for waypoint_ind in range(len(targets)):
        targets[waypoint_ind] += '%s' %(waypoint_ind // len(g_conf.TARGETS))
    
    g_conf.TARGETS = targets

    print('Types of outputs: %s' %g_conf.TARGET_KEYS)
    print('All targets: %s' %g_conf.TARGETS)


def _get_weights_for_target_losses():
    # weighting loss for each waypoint
    if g_conf.WAYPOINT_LOSS_WEIGHT == 'exponential':
        g_conf.WAYPOINT_LOSS_WEIGHT = np.exp( np.arange(g_conf.NUMBER_OF_WAYPOINTS)*-1 )
    else:
        g_conf.WAYPOINT_LOSS_WEIGHT = np.ones( g_conf.NUMBER_OF_WAYPOINTS )


def _get_model_configuration_flags():

    use_seg_output = 'segmentation_head' in g_conf.MODEL_CONFIGURATION['branches'].keys() and \
                    g_conf.MODEL_CONFIGURATION['branches']['segmentation_head'] == 1
    use_seg_input = 'seg_input' in g_conf.MODEL_CONFIGURATION.keys() and \
                    g_conf.MODEL_CONFIGURATION['seg_input']['activate'] == 1

    # check for impossible model configurations
    if use_seg_input and use_seg_output:
        print('\nInvalid model type: seg input or seg output, choose one pls')
        exit()
    
    fusion_type = None

    if 'seg_input' in g_conf.MODEL_CONFIGURATION.keys():
        fusion_type = g_conf.MODEL_CONFIGURATION['seg_input']['type']
        if use_seg_input and fusion_type != 'EF' and fusion_type != 'MF' and fusion_type != 'SS':
            print('\nInvalid fusion type: %s even though seg_input is active. expect EF or MF or SS' %fusion_type)
            exit()
        if g_conf.USE_PATHING and ('steer' in g_conf.TARGET_KEYS or 'throttle' in g_conf.TARGET_KEYS):
            print('\nInvalid target: USE_PATHING is true, pathing only uses angle, no steer, throttle or brake!')
            exit()

    return(fusion_type, use_seg_input, use_seg_output)


def _output_seg_mask_to_tensorboard(seg_mask_tensor, iteration, name, color_map=None):
    
    segmentation_output = torch.squeeze(seg_mask_tensor).cpu().detach().numpy()
    segmentation_output = np.argmax(segmentation_output, axis=1)
    
    segmentation_rgb = np.zeros((segmentation_output.shape[0], segmentation_output.shape[1], 
                       segmentation_output.shape[2], 3))

    for i in range(segmentation_output.shape[0]):
        segmentation_rgb[i] = color_map[segmentation_output[i]]
    
    # transpose is taken care of in coil logger
    coil_logger.add_image(name, torch.from_numpy(segmentation_rgb).permute(0,3,1,2), iteration)


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, suppress_output=True, number_of_workers=12):
    """
        The main training function. This functions loads the latest checkpoint
        for a given, exp_batch (folder) and exp_alias (experiment configuration).
        With this checkpoint it starts from the beginning or continue some training.
    Args:
        gpu: The GPU number
        exp_batch: the folder with the experiments
        exp_alias: the alias, experiment name
        suppress_output: if the output are going to be saved on a file
        number_of_workers: the number of threads used for data loading

    Returns:
        None

    """
    try:
        # We set the visible cuda devices to select the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        g_conf.VARIABLE_WEIGHT = {}
        # At this point the log file with the correct naming is created.
        # You merge the yaml file with the global configuration structure.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))
        
        _get_targets()
        _get_weights_for_target_losses()
        fusion_type, use_seg_input, use_seg_output = _get_model_configuration_flags()

        set_type_of_process('train')
        # Set the process into loading status.
        coil_logger.add_message('Loading', {'GPU': gpu})

        # Put the output to a separate file if it is the case

        if suppress_output:
            if not os.path.exists('_output_logs'):
                os.mkdir('_output_logs')
            sys.stdout = open(os.path.join('_output_logs', exp_alias + '_' +
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a",
                              buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_'+g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)

        if coil_logger.check_finish('train'):
            coil_logger.add_message('Finished', {})
            return

        # Preload option
        if g_conf.PRELOAD_MODEL_ALIAS is not None:
            checkpoint = torch.load(os.path.join('_logs', g_conf.PRELOAD_MODEL_BATCH,
                                                  g_conf.PRELOAD_MODEL_ALIAS,
                                                 'checkpoints',
                                                 str(g_conf.PRELOAD_MODEL_CHECKPOINT)+'.pth'))

        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint_file = get_latest_saved_checkpoint()
        if checkpoint_file is not None:
            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias,
                                    'checkpoints', str(get_latest_saved_checkpoint())))
            iteration = checkpoint['iteration']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']
        else:
            iteration = 0
            best_loss = 10000.0
            best_loss_iter = 0


        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        # By instantiating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        # Instantiate the class used to read a dataset. The coil dataset generator
        # can be found
        if use_seg_output or use_seg_input:
            dataset = CoILDatasetWithSeg(full_dataset, transform=augmenter,
                    preload_name=str(g_conf.NUMBER_OF_HOURS)
                    + 'hours_withseg_' + g_conf.TRAIN_DATASET_NAME)
        elif g_conf.USE_PATHING:
            dataset = CoILDatasetWithPathing(full_dataset, transform=augmenter,
                    preload_name=str(g_conf.NUMBER_OF_HOURS)
                    + 'hours_withpath_' + g_conf.TRAIN_DATASET_NAME)
        else: 
            dataset = CoILDatasetWithWaypoints(full_dataset, transform=augmenter,
                    preload_name=str(g_conf.NUMBER_OF_HOURS)
                    + 'hours_' + g_conf.TRAIN_DATASET_NAME)
        print("Loaded dataset")

        data_loader = select_balancing_strategy(dataset, iteration, number_of_workers)
        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE)

        if checkpoint_file is not None or g_conf.PRELOAD_MODEL_ALIAS is not None:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            accumulated_time = checkpoint['total_time']
            loss_window = coil_logger.recover_loss_window('train', iteration)
        else:  # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            loss_window = []

        print("Before the loss")

        criterion = Loss(g_conf.LOSS_FUNCTION)

        # Loss time series window
        for data in data_loader:

            # Basically in this mode of execution, we validate every X Steps, if it goes up 3 times,
            # add a stop on the _logs folder that is going to be read by this process
            if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
                    check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
                break
            """
                ####################################
                    Main optimization loop
                ####################################
            """

            iteration += 1
            if iteration % 1000 == 0:
                adjust_learning_rate_auto(optimizer, loss_window)

                # log learning rate into tensorboard
                for param_group in optimizer.param_groups:
                    logged_lr = param_group['lr']
                    break

                coil_logger.add_scalar('Learning rate', logged_lr, iteration)

            # get the control commands from float_data, size = [120,1]

            capture_time = time.time()
            controls = data['directions']
            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
            model.zero_grad()

            if fusion_type == 'SS':
                model_input_img = None
            else:
                model_input_img = torch.squeeze(data['rgb'].cuda())
                
            if use_seg_input:
                model_input_seg = dataset.extract_seg_gt(data).cuda()
            else:
                model_input_seg = None
            
            branches, vis = model(model_input_img,
                                model_input_seg,
                                dataset.extract_inputs(data).cuda())

            loss_function_params = {
                'branches': branches,
                'targets': dataset.extract_targets(data).cuda(),
                'controls': controls.cuda(),
                'inputs': dataset.extract_inputs(data).cuda(),
                'branch_weights': g_conf.BRANCH_LOSS_WEIGHT,
                'variable_weights': g_conf.VARIABLE_WEIGHT,
                'use_seg_output': use_seg_output,
                'ignore': dataset.extract_ignore_waypoint_mask(data).cuda()
            }
            if use_seg_output and not use_seg_input:
                loss_function_params['seg_ground_truth'] = dataset.extract_seg_gt(data).cuda()
            loss, _ = criterion(loss_function_params)
            loss.backward()
            optimizer.step()
            """
                ####################################
                    Saving the model if necessary
                ####################################
            """

            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'optimizer': optimizer.state_dict(),
                    'best_loss_iter': best_loss_iter
                }
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                               , 'checkpoints', str(iteration) + '.pth'))

            """
                ################################################
                    Adding tensorboard logs.
                    Making calculations for logging purposes.
                    These logs are monitored by the printer module.
                #################################################
            """
            coil_logger.add_scalar('Loss', loss.data, iteration)
            rgb_image = torch.squeeze(data['rgb'])
            coil_logger.add_image('Image', rgb_image, iteration)

            if use_seg_output:
                _output_seg_mask_to_tensorboard(branches[-1], iteration, name = 'Segmentation Output', color_map = carla_cm)
                vis = loss_function_params['seg_ground_truth']
            if vis is not None:
                _output_seg_mask_to_tensorboard(vis, iteration, name = 'Segmentation Input', color_map = carla_cm)
            
            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration

            # Log a random position
            position = random.randint(0, g_conf.BATCH_SIZE-1)

            output = model.extract_branch(torch.stack(branches[0:4]), controls)
            error = torch.abs(output - dataset.extract_targets(data).cuda())

            accumulated_time += time.time() - capture_time

            coil_logger.add_message('Iterating',
                                    {'Iteration': iteration,
                                     'Loss': loss.data.tolist(),
                                     'Images/s': (iteration * g_conf.BATCH_SIZE) / accumulated_time,
                                     'BestLoss': best_loss, 'BestLossIteration': best_loss_iter,
                                     'Output': output[position].data.tolist(),
                                     'GroundTruth': dataset.extract_targets(data)[
                                         position].data.tolist(),
                                     'Error': error[position].data.tolist(),
                                     'Inputs': dataset.extract_inputs(data)[
                                         position].data.tolist()},
                                    iteration)
            loss_window.append(loss.data.tolist())
            coil_logger.write_on_error_csv('train', loss.data)
            
            if iteration % 10 == 0:
                print("Iteration: %d  Loss: %f" % (iteration, loss.data))

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

#    except RuntimeError as e:

#        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
