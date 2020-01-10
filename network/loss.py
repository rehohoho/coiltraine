from . import loss_functional as LF
import torch
from configs import g_conf
import numpy as np

def l1(params):
    return branched_loss(LF.l1_loss, params)


def l2(params):
    return branched_loss(LF.l2_loss, params)


def l1_attention(params):
    return branched_loss(LF.l1_attention_loss, params)


def branched_loss(loss_function, params):

    """
    Args
        loss_function: The loss functional that is actually computing the loss
        params: all the parameters, including
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls: the controls used for each point
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                variable_weights: The weights for each of the variables used

                For other losses it could contain more parameters

    Returns
        The computed loss function, but also a dictionary with plotable variables for tensorboard
    """

    controls_mask = LF.compute_branches_masks(params['controls'],
                                              params['branches'][0].shape[1])
    # Update the dictionary to add also the controls mask.
    params.update({'controls_mask': controls_mask})

    # calculate loss for each branch with specific activation
    loss_branches_vec, plotable_params = loss_function(params)

    # Apply the variable weights
    # This is applied to all branches except the last one, that is the speed branch...
    # TODO This is hardcoded to have 4 branches not using speed.
    
    n_waypoints = g_conf.NUMBER_OF_WAYPOINTS
    n_outputs = len(g_conf.TARGET_KEYS)
    loss_branches_vec_len = n_outputs * n_waypoints

    for i in range(4):
        
        for waypoint in range(n_waypoints):
            waypoint_ind = waypoint*n_outputs
            for output in range(n_outputs):
                loss_branches_vec[i][:, waypoint_ind+output] = loss_branches_vec[i][:, waypoint_ind+output] \
                                        * params['variable_weights'][g_conf.TARGET_KEYS[output]] \
                                        * g_conf.WAYPOINT_LOSS_WEIGHT[waypoint]
        
        loss_branches_vec[i] *= params['ignore']
        loss_branches_vec[i] = torch.sum(loss_branches_vec[i], dim=1)
    
    loss_function = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + \
                    loss_branches_vec[3]

    loss_function = loss_function/ np.sum(g_conf.WAYPOINT_LOSS_WEIGHT)     # adjust magnitiude of branch loss to be roughly comparable to speed loss

    speed_loss = loss_branches_vec[4] / (params['branches'][0].shape[0])
    
    loss = torch.sum(loss_function) / (params['branches'][0].shape[0])\
        + torch.sum(speed_loss) / (params['branches'][0].shape[0])

    if params['use_seg_output']:
        seg_loss = loss_branches_vec[5] / (params['branches'][0].shape[0])
        loss += torch.sum(seg_loss) / (params['branches'][0].shape[0])
        
    return loss, plotable_params


def Loss(loss_name):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if loss_name == 'L1':

        return l1

    elif loss_name == 'L2':

        return l2

    else:
        raise ValueError(" Not found Loss name")


