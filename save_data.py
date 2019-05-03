""" 
Create basicdataset from gameplay. After launching the program, 
press ` key(grave) to start recording and press again to stop
the program. Images and key records are saved with begin time
as name.
"""
from screen_logger import screen_logger
from keyboard_logger import keyboard_logger
from info import info
from time import sleep
import datetime
import cv2
import os

def main():
    # Initialize data. Screenshot position and size can be changed.
    slog = screen_logger(top=info.top, left=info.left, width=info.width, height=info.height)
    klog = keyboard_logger()
    name = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    os.mkdir(name)
    loc = name + '/' + name + '_'

    # Wait until ` key(grave) is pressed.
    while not klog.get_key_pressed(['grave'])[0]:
        sleep(0.1)
    
    # Capture screenshot, key press and save until stopped
    i = 0
    with open(loc + 'key.txt', 'w') as kf:
        while True:
            keys = klog.get_key_pressed(['space', 'grave'])
            # If ` key is pressed, stop the program
            if keys[1]:
                break
            raw_image = slog.get_capture()
            cv2.imwrite(loc + '{}.png'.format(i), raw_image[:,:,0])
            kf.write('1' if keys[0] else '0')
            i += 1

    klog.stop()

if __name__ == "__main__":
    main()
