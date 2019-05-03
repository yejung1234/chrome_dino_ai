""" 
Running this script trains model and save. Main function should 
be modified according to needs.
"""
import numpy as np
from dinomodel import dinomodel
from info import info
import cv2

def dinoset_load(name, time_batch, height, width):
    """Load dataset and format it for training
    
    Args:
        name (str): Folder and file name, probably in format %y%m%d_%H%M%S
        time_batch (int): Number of channels for each dataset input
        height (int): Height for preprocessed image
        width (int): Width for preprocessed image
    
    Returns:
        tuple: First is features, second is labels
    """
    # Load key input data.
    longname = name + '/' + name + '_'
    with open(longname +'key.txt') as keyfile:
        original_keys = keyfile.readline()
        if original_keys[-1] == '\n':
            original_keys = original_keys[:-1]
        original_keys = np.array(list(original_keys)).astype(np.float32)
    size = len(original_keys)

    # Modify labels so that a frame before actual key input also have label 1.
    # The reason is that such frames can also be good timing for jumping.
    keys = np.zeros(size)
    for i in range(2):
        keys[max(0, i-1):min(size, size-1+i)] += original_keys[max(0, 1-i):min(size, size+1-i)]
    keys = keys.reshape(size, 1)[time_batch - 1:]

    # Load images and arrange as dataset.
    pictures = np.ndarray([size, height, width], dtype=np.float32)
    for i in range(size):
        pictures[i] = cv2.imread(longname + str(i) + '.png', 0).astype(np.float32)
    big_pictures = np.expand_dims(np.arange(size-time_batch+1), 1)
    big_pictures = big_pictures + np.arange(time_batch)
    big_pictures = pictures[big_pictures].swapaxes(1, 2).swapaxes(2, 3)
    
    return big_pictures, keys

def main():
    # Number of frames used in one dataset.
    # Default is 4 frames to determine if one should jump or not.
    time_batch = info.time_batch
    model = dinomodel(info.height, info.width, time_batch, info.learning_rate)

    # Saved model name
    model_name = info.model_path

    # Load dataset.
    dataset1 = dinoset_load('190411_162151', time_batch, model.fixed_height, model.fixed_width)
    dataset2 = dinoset_load('190416_203057', time_batch, model.fixed_height, model.fixed_width)
    dataset3 = dinoset_load('190418_194047', time_batch, model.fixed_height, model.fixed_width)
    features = np.concatenate((dataset1[0], dataset2[0], dataset3[0]), 0)
    labels = np.concatenate((dataset1[1], dataset2[1], dataset3[1]), 0)

    # Split dataset by labels.
    index0 = np.where(labels == 0)[0]
    index1 = np.where(labels == 1)[0]
    one_size = index1.size

    # Try to restore model checkpoint if is available.
    try:
        model.restore_model(model_name)
    except:
        print('Failed to load model.\nStart training with random variables.')

    with open('train.log', 'a') as log:
        try:
            for i in range(10000):
                # Randomly select same amount of data with label 0 as with label 1.
                # If we use all data, there will be too many data with label 0
                # and model will try to predict every cases to 0.
                rand_index0 = np.random.permutation(index0)[:one_size]
                indices = np.concatenate((rand_index0, index1))

                loss = model.train(features[indices], labels[indices])
                raw_prediction = model.predict(features)
                msg = 'Iteration #{} : loss {} /'.format(i+1, loss)
                
                # Calculate f1 score for evaluating model.
                for j in range(1, 10):
                    prediction = raw_prediction >= (j/10)
                    TP = (prediction == labels).sum()
                    FP = (prediction == 1-labels).sum()
                    FN = (1-prediction == labels).sum()
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    f1score = 2*precision*recall / (precision+recall)
                    msg += ' f1{} {} /'.format(j, np.floor(f1score*1000)/1000)
                print(msg)
                log.write(msg + '\n')
                
                # Stop if loss becomes small enough.
                if loss < 0.05:
                    break
        except Exception as e:
            print(e)

    # Make checkpoint of model.
    model.save_model(model_name)
    
if __name__ == '__main__':
    main()