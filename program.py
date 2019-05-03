""" Actual program to observe game and decide whether to jump or not.
"""
from screen_logger import screen_logger
from keyboard_logger import keyboard_logger
from dinomodel import dinomodel
from time import sleep
from pynput.keyboard import Key, Controller
from info import info
import numpy as np
import os
import cv2
import datetime

def main():
    # Define objects
    slog = screen_logger(top=info.top, left=info.left, width=info.width, height=info.height)
    time_batch = info.time_batch
    klog = keyboard_logger()
    kcont = Controller()
    model = dinomodel(info.height, info.width, time_batch, info.learning_rate)
    model.restore_model(info.model_path)
    
    # Wait until ` key is pressed
    print('\n\nReady to go\n')
    while not klog.get_key_pressed(['grave'])[0]:
        sleep(0.1)
    
    pictures = np.ndarray((1, model.fixed_height, model.fixed_width, time_batch))
    i = 0
    image_list = []
    while True:
        keys = klog.get_key_pressed(['space', 'grave'])
        # If ` key is pressed again, stop program
        if keys[1]:
            break
        
        # Get image ready
        raw_image = slog.get_capture()[:,:,0].reshape(1, info.height, info.width, 1)
        small_image = model.preprocess(raw_image)

        # Move images to one frame front and put the new image at the end
        pictures[:,:,:,0:time_batch-1] = pictures[:,:,:,1:time_batch]
        pictures[:,:,:,time_batch-1:time_batch] = small_image
        image_list.append(raw_image[0])

        # If program has just started, wait until images are ready
        if i < time_batch:
            i += 1
        else:
            prediction = model.predict(pictures)
            print(prediction)
            # If prediction is big enough, press jump key
            if prediction >= 0.5:
                kcont.press(Key.space)
                kcont.release(Key.space)
    klog.stop()

if __name__ == "__main__":
    main()

