""" 
This module contains helper functions for experimental calibrations.

This module will require future updates for other behaviours.

"""


def pixels_to_cm(wall='2mm_basic'):
    """Simple call to get pixel conversion to centimeters for pose estimation data. This function will be updated in the future for other behaviour experiments.

    Args:
        wall (str, optional): Experimental mode. Defaults to '2mm_basic'.

    Returns:
        float: Conversion between pixels and centimeters
    """

    # wall distance 6mm average = 25.3px
    # wall distance 6mm std = 1.6px
    if wall == '2mm_basic':
        px_to_cm = (6.0/25.3)*0.1 #ratio to multiply pixels by
    return px_to_cm