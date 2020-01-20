
from time import sleep
import argparse
import os
import numpy as np

from imutils.video import VideoStream, FPS
import cv2
import torch
import serial

from configs import g_conf, merge_with_yaml
from network import CoILModel


def raw_image_to_tensor(img):
    """ Change cv2 image to tensor of right shape to be fed to model """
    
    rgb_input = cv2.resize(img, (200, 88), interpolation = cv2.INTER_AREA)
    rgb_input = rgb_input.astype(np.float)
    rgb_input = torch.from_numpy(rgb_input).type(torch.FloatTensor)
    rgb_input = rgb_input.unsqueeze(0).transpose(2,3).transpose(1,2)
    rgb_input /= 255

    return(rgb_input)


def add_text_to_frame(frame, steer, throttle, brake):
    frame = cv2.putText(frame, 
                        'Steer: %s' %steer,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    frame = cv2.putText(frame, 
                        'Throttle: %s' %throttle,
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    frame = cv2.putText(frame, 
                        'Brake: %s' %brake,
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return(frame)


def key_to_controls(key, control, speed):

    new_control = control
    new_speed = speed

    if key == ord('q'):
        print('quitting...')
        break
    if key == ord('w'):
        print('going forward')
        control = 5.0
    if key == ord('a'):
        print('going left')
        control = 3.0
    if key == ord('d'):
        print('going right')
        control = 4.0
    if key == ord('f'):
        print('follow lane')
        control = 2.0
    
    if key == ord('-'):
        print('decreasing speed')
        new_speed -= 10
    if key == ord('='):
        print('increasing speed')
        new_speed += 10
    if key == ord('s'):
        print('stop')
        new_speed = 0
    
    return( control, new_speed )
    

def main(output_path, checkpoint_file, config_file):
    """ Executor
    
    Args:
        output_path         folder to save images to if required
        checkpoint_file     pth holding pytorch CIL model
        config_file         yaml holding CIL config used for checkpoint_file

    controls:
        REACH_GOAL = 0.0
        
        # only at intersections
        GO_STRAIGHT = 5.0
        TURN_RIGHT = 4.0
        TURN_LEFT = 3.0

        # when no intersection
        LANE_FOLLOW = 2.0
    """

    try:
        print("Starting the video stream...")
        vs = VideoStream(src=0).start()
        sleep(2)
        fps = FPS().start()

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        # setting up model
        merge_with_yaml(config_file)
        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        model.cuda()

        # loading checkpoint into model
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])

        # set up serial for arduino
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        speed = 0.0
        control = 2.0

        while True:
            
            frame = vs.read()
            reading = ser.readline().decode('utf-8')
            if reading[0] == '<' and reading[-3] == '>':
                speed = float( reading[1:-3].split(',')[1] )

            # inputs
            rgb_input = raw_image_to_tensor(frame).cuda()
            seg_input = None
            speed_input = torch.tensor([[speed]]).cuda()
            control_input = torch.tensor([control])

            # processing through model
            branches = model(rgb_input, seg_input, speed_input)
            output = model.extract_branch(torch.stack(branches[0:4]), control_input)
            steer, throttle, brake = output.data.tolist()[0]

            # add outputs to frame and show
            frame = add_text_to_frame(frame, steer, throttle, brake)
            cv2.imshow('Frame', frame)
            
            # output to arduino
            ser.write( b'<%03d%03d>' %(steer * 55 + 55, 30) )

            # check when to break loop or alter controls
            key = cv2.waitKey(1) & 0xFF
            control, speed = key_to_controls(key, control, speed)

            fps.update()

    except Exception as e:
        print(e)

    # clean up
    ser.write( b'<%03d%03d>' %(55, 0) )
    fps.stop()
    print("Elapsed %s, approx FPS %s" %(fps.elapsed(), fps.fps()) )

    cv2.destroyAllWindows()
    print('destroyed windows...')
    vs.stop()
    print('stream stopped...')


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--output_folder',
        required=True,
        type=str,
        dest='output_folder'
    )
    argparser.add_argument(
        '--checkpoint_file',
        required=True,
        type=str,
        dest='checkpoint_file'
    )
    argparser.add_argument(
        '--config_file',
        required=True,
        type=str,
        dest='config_file'
    )

    args = argparser.parse_args()
    main(args.output_folder, args.checkpoint_file, args.config_file)