from lane import *
from preprocess import *
def p4_init(root=None, file=None):
    load_data(root)

    M_max = parameters['M_max']
    MInv_max = parameters['MInv_max']
    M_mid = parameters['M_mid']
    MInv_mid = parameters['MInv_mid']
    M_min = parameters['M_min']
    MInv_min = parameters['MInv_min']
    parameters['p'] = False
    parameters['b'] = False

    f1 = 'project_video.mp4'
    f2 = 'challenge_video.mp4'
    f3 = 'harder_challenge_video.mp4'

    # Change Input File Here

    input_file = file
    if input_file == f3:
        parameters['M'] = M_mid
        parameters['MInv'] = MInv_mid
        parameters['color_sw'] = False
        parameters['x'] = True
        parameters['y'] = True
        parameters['m'] = False
        parameters['d'] = False
        parameters['margin'] = 65
    else:
        parameters['M'] = M_max
        parameters['MInv'] = MInv_max
        parameters['color_sw'] = True
        parameters['use_color'] = True
        parameters['x'] = False
        parameters['y'] = False
        parameters['m'] = False
        parameters['d'] = False
        parameters['margin'] = 40

    lane = Lane(parameters)
    lane_verify = Lane(parameters)
    return lane, lane_verify, pipeline
