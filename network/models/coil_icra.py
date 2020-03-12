from logger import coil_logger
import torch.nn as nn
import torch
import importlib

from configs import g_conf
from coilutils.general import command_number_to_index

from .building_blocks import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join
from .building_blocks import SegmentationBranch

class CoILICRA(nn.Module):

    def __init__(self, params):
        # TODO: Improve the model autonaming function

        super(CoILICRA, self).__init__()
        self.params = params

        # if using 2 channel bin-mask only
        if 'seg_input' in params.keys() and params['seg_input']['activate'] and params['seg_input']['ridable_class'] != -1:
            params['seg_input']['n_classes'] = 2

        # handles SS and EF, conv and resnet
        number_output_neurons = self._init_rgb_encoder(params)
        # handles MF and no MF
        join_neurons = self._init_seg_encoder(params, number_output_neurons)

        self.measurements = FC(params={'neurons': [len(g_conf.INPUTS)] +
                                                   params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})
        self.join = Join(
            params={'after_process':
                         FC(params={'neurons':
                                        join_neurons +
                                        params['join']['fc']['neurons'],
                                     'dropouts': params['join']['fc']['dropouts'],
                                     'end_layer': False}),
                     'mode': 'cat'
                    }
         )
        self.speed_branch = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                  params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})
        if 'segmentation_head' in params['branches'].keys() and params['branches']['segmentation_head']:
            self.segmentation_branch = SegmentationBranch()

        # Create the fc vector separatedely
        branch_fc_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                         params['branches']['fc']['neurons'] +
                                                         [len(g_conf.TARGETS)],
                                               'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                                               'end_layer': True}))

        self.branches = Branching(branch_fc_vector)  # Here we set branching automatically

        if 'conv' in params['perception']:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)


    def forward(self, rgb, seg_mask, a):
        
        """ ###### APPLY THE SEGMENTATION INPUT MODULE """
        # check for seg_mask to exist is done in main train script
        if seg_mask is not None:
            
            # get single channel mask
            # mask = torch.argmax( seg_mask.transpose(1,2).transpose(2,3), 3 )
            # mask = mask.type(torch.cuda.FloatTensor)
            # mask = torch.unsqueeze(mask, -3)

            if g_conf.MODEL_CONFIGURATION['seg_input']['ridable_class'] != -1:
                # set mask to 2 channels only, ridable and non-ridable areas
                ridable_class = g_conf.MODEL_CONFIGURATION['seg_input']['ridable_class']
                ridable = seg_mask[:, ridable_class]
                non_ridable = seg_mask.sum(dim = 1) - ridable
                seg_mask = torch.stack((ridable, non_ridable), dim = 1)

            if g_conf.MODEL_CONFIGURATION['seg_input']['type'] == 'MF':
                
                # pad single channel to get three channels
                # seg_mask = torch.nn.functional.pad( mask, pad = (0,0,0,0,0,2), value = 0)

                encoded_seg, seg_inter = self.seg_perception(seg_mask)
                
                # self.seg_intermediate_layers = seg_inter # intermediate layers for future vizualization
                
            if g_conf.MODEL_CONFIGURATION['seg_input']['type'] == 'EF':
                # add mask channel to rgb image
                rgb = torch.cat( (rgb, seg_mask), -3 )

        """ ###### APPLY THE PERCEPTION MODULE """
        if rgb is not None:
            encoded_rgb, inter = self.perception(rgb) # return x, [x0, x1, x2, x3, x4]
        # only for SS
        else:
            encoded_rgb, inter = self.perception(encoded_seg)

        # self.intermediate_layers = inter # intermediate layers for future vizualization

        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)
        
        """ Join measurements and perception"""
        if g_conf.MODEL_CONFIGURATION['seg_input']['type'] == 'MF':
            j = self.join(encoded_rgb, m, encoded_seg)
        else:
            j = self.join(encoded_rgb, m)

        branch_outputs = self.branches(j)

        speed_branch_output = self.speed_branch(encoded_rgb)

        # We concatenate speed with the rest.
        branch_outputs += [speed_branch_output]
        if 'segmentation_head' in self.params['branches'].keys() and self.params['branches']['segmentation_head']:
            seg_map = self.segmentation_branch(inter)
            branch_outputs += [seg_map]

        # branch_outputs += [seg_mask]

        return branch_outputs


    def forward_branch(self, rgb, x_seg, a, branch_number):
        """
        DO a forward operation and return a single branch.

        Args:
            rgb:            the image input
            a:              speed measurement
            branch_number:  the branch number to be returned

        Returns:
            the forward operation on the selected branch

        """
        # Convert to integer just in case .
        # TODO: take four branches, this is hardcoded
        output_vec = torch.stack(self.forward(rgb, x_seg, a)[0:4])

        return self.extract_branch(output_vec, branch_number)


    def get_perception_layers(self, rgb):
        return self.perception.get_layers_features(rgb)


    # used for generating output to logs
    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]


    def _init_rgb_encoder(self, params):

        number_first_layer_channels = 0

        for _, sizes in g_conf.SENSORS.items():
            number_first_layer_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        # For this case we check if the perception layer is of the type "conv"
        if 'conv' in params['perception']:
            perception_convs = Conv(params={'channels': [number_first_layer_channels] +
                                                          params['perception']['conv']['channels'],
                                            'kernels': params['perception']['conv']['kernels'],
                                            'strides': params['perception']['conv']['strides'],
                                            'dropouts': params['perception']['conv']['dropouts'],
                                            'end_layer': True})

            perception_fc = FC(params={'neurons': [perception_convs.get_conv_output(sensor_input_shape)]
                                                  + params['perception']['fc']['neurons'],
                                       'dropouts': params['perception']['fc']['dropouts'],
                                       'end_layer': False})

            self.perception = nn.Sequential(*[perception_convs, perception_fc])

            number_output_neurons = params['perception']['fc']['neurons'][-1]

        elif 'res' in params['perception']:  # pre defined residual networks
            
            input_channels = 3  # RGB only
            if 'seg_input' in params.keys() and params['seg_input']['activate']:
                # RGB + binmask for n_classes
                if g_conf.MODEL_CONFIGURATION['seg_input']['type'] == 'EF':
                    input_channels = 3 + params['seg_input']['n_classes']
                # binmask for n_classes only
                if g_conf.MODEL_CONFIGURATION['seg_input']['type'] == 'SS':
                    input_channels = params['seg_input']['n_classes']

            resnet_module = importlib.import_module('network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, params['perception']['res']['name'])
            self.perception = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                            input_channels=input_channels,
                                            num_classes=params['perception']['res']['num_classes'])

            number_output_neurons = params['perception']['res']['num_classes']

        else:

            raise ValueError("invalid convolution layer type")

        return number_output_neurons
    

    def _init_seg_encoder(self, params, number_output_neurons):
        
        # Mid fusion appends segmentation encoded information onto rgb encoded information
        if 'seg_input' in params.keys() and params['seg_input']['activate'] and params['seg_input']['type'] == 'MF':
            input_channels = params['seg_input']['n_classes']

            resnet_module = importlib.import_module('network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, params['seg_input']['res']['name'])
            self.seg_perception = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                            input_channels=input_channels,
                                            num_classes=params['seg_input']['res']['num_classes'])

            join_neurons = [params['measurements']['fc']['neurons'][-1] +
                            params['seg_input']['res']['num_classes'] +
                            number_output_neurons]
        else:
            join_neurons = [params['measurements']['fc']['neurons'][-1] +
                            number_output_neurons]
        

        return join_neurons