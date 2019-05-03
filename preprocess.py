""" 
Preprocess to simplify images. First, if background of image is
bright, invert colors of image. This way, both night and day in-
game will have same image representation. Second, use max poolin
g multiple times to make image smaller, while human brain can st
ill handle images and figure out whether to press the key or not
. After preprocessing, save the images with same name.
"""
from dinomodel import dinomodel
from info import info
import cv2
import numpy as np

def main():
    name = input('dataset name : ')

    # Figure out the number of images in dataset.
    with open('{}/{}_key.txt'.format(name, name)) as f:
        l = len(f.readline())

    # Load images into array with shape [l, height, width, 1]
    images = np.ndarray((l, info.height, info.width))
    for i in range(l):
        images[i] = cv2.imread('{}/{}_{}.png'.format(name, name, i), 0).astype(np.float32)
    images = np.expand_dims(images, 3)
    
    # Process images
    model = dinomodel(info.height, info.width, info.time_batch, info.learning_rate)
    images = model.preprocess(images)

    # Write back images
    images = np.squeeze(images).astype(np.uint8)
    for i in range(l):
        cv2.imwrite('{}/{}_{}.png'.format(name, name, i), images[i])

if __name__ == "__main__":
    main()