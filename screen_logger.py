""" Defines helper class for getting screenshot image.
"""
import numpy as np
import cv2
from mss import mss


class screen_logger():
    """ Helper class for screenshot.
    """    
    def __init__(self, top, left, height, width):
        """Creates new screenshot wrapper with given numbers
        
        Args:
            top (int): pixel offset of top-left point of screenshot area from top of screen
            left (int): pixel offset of top-left point of screenshot area from left of screen
            height (int): height of screenshot area in pixels
            width (int): width of screenshot area in pixels
        """
        self.dims = {'top':top, 'left':left, 'height':height, 'width':width}
        self.sct = mss()

    def get_capture(self):
        """Get screen capture from given area
        
        Returns:
            numpy array: 2D array of uint8 representing image
        """
        ss = self.sct.grab(self.dims)
        return np.array(ss.pixels).astype(np.uint8)

