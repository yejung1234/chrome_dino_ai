"""Defines chrome dino game deep learning model and preprocessing model
"""
import tensorflow as tf
import numpy as np
import cv2

class dinomodel():
    """Chrome dino game deep learning model and preprocessing model
    """
    def __init__(self, raw_height, raw_width, time_batch, learning_rate):
        """Defines pooling/learning tensorflow graph
        
        Args:
            raw_height (int): Height of original screenshot
            raw_width (int): Width of original screenshot
            time_batch (int): Number of channels of one data, each channel representing state at different time
            learning_rate (float): Initial learning rate of adam optimizer
        """
        tf.reset_default_graph()
        sess = tf.Session()
        
        pooling_in_shape = [None, raw_height, raw_width, 1]
        
        pooling_in = tf.placeholder(dtype=np.float32, shape=pooling_in_shape)
        img_mean = tf.reduce_mean(pooling_in, [1, 2], keepdims=True)
        img_max = tf.reduce_max(pooling_in, [1, 2], keepdims=True)
        img_min = tf.reduce_min(pooling_in, [1, 2], keepdims=True)
        pooling_norm = (pooling_in - img_min) / (img_max - img_min) * 255
        is_afternoon = tf.cast(img_mean >= 127.5, float) 
        pooling_0 = (is_afternoon) * (255-pooling_norm) + (1 - is_afternoon) * pooling_norm
        pooling_1 = tf.nn.max_pool(pooling_0, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        pooling_2 = tf.nn.max_pool(pooling_1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        pooling_3 = tf.nn.max_pool(pooling_2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        pooling_4 = tf.nn.max_pool(pooling_3, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        
        [_, fixed_height, fixed_width, _] = pooling_4.shape.as_list()
        model_in_shape = [None, fixed_height, fixed_width, time_batch]

        input_pictures = tf.placeholder(np.float32, shape=model_in_shape, name='input_pictures')
        label_key = tf.placeholder(np.float32, shape=(None, 1), name='label_key')
        conv1v = tf.Variable(np.random.normal(size=[5, 5, time_batch, time_batch * 2]), dtype=np.float32)
        conv1 = tf.nn.conv2d(input_pictures / 255, conv1v, [1, 1, 1, 1], 'SAME')
        conv1b = tf.Variable(np.random.normal(size=[i if i != None else 1 for i in conv1.shape.as_list()]), dtype=np.float32)
        conv1s = tf.nn.relu(conv1 + conv1b)
        conv2v = tf.Variable(np.random.normal(size=[5, 5, time_batch * 2, time_batch * 4]), dtype=np.float32)
        conv2 = tf.nn.conv2d(conv1s, conv2v, [1, 1, 1, 1], 'SAME')
        conv2b = tf.Variable(np.random.normal(size=[i if i != None else 1 for i in conv2.shape.as_list()]), dtype=np.float32)
        conv2s = tf.nn.relu(conv2 + conv2b)
        flat = tf.keras.layers.Flatten()(conv2s)
        dense_1 = tf.keras.layers.Dense(256, activation='sigmoid', kernel_initializer='RandomNormal',name='dense_1')(flat)
        dense_2 = tf.keras.layers.Dense(256, activation='sigmoid', kernel_initializer='RandomNormal',name='dense_2')(dense_1)
        result = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='RandomNormal',name='result')(dense_2)
        loss = -tf.reduce_mean(label_key * tf.log(result) + (1 - label_key) * tf.log(1 - result))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update = optimizer.minimize(loss, name='update')
        
        self.sess = sess
        self.raw_height = raw_height
        self.raw_width = raw_width
        self.pool_in = pooling_in
        self.pool_out = pooling_4
        self.fixed_height = fixed_height
        self.fixed_width = fixed_width
        self.time_batch = time_batch
        self.model_in = input_pictures
        self.model_out = result
        self.model_label = label_key
        self.model_loss = loss
        self.update = update
        self.saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        
    def save_model(self, name):
        """Save current model's variables to file
        
        Args:
            name (str): Path to save model
        """
        self.saver.save(self.sess, name)
        
    def restore_model(self, name):
        """Restore variables from saved file
        
        Args:
            name (str): Path to load model
        """
        self.saver.restore(self.sess, name)
        
    def preprocess(self, image):
        """Process raw image to get smaller and remove difference between day and night in game
        
        Args:
            image (numpy array): Numpy array with shape (batch, height, width, 1) representing images
        
        Returns:
            numpy array: Numpy array with (batch, small_height, small_width, 1) representing preprocessed images
        """
        feed_dict = {self.pool_in:image}
        return self.sess.run(self.pool_out, feed_dict=feed_dict)
    
    def train(self, images, keys):
        """Train model with given images and keys
        
        Args:
            images (numpy array): Numpy array with (batch, small_height, small_width, time_batch), features of dataset
            keys (list): List of 0 or 1, labels of dataset
        
        Returns:
            float: Loss of training
        """
        feed_dict = {self.model_in:images, self.model_label:keys}
        loss = self.sess.run(self.model_loss, feed_dict)
        self.sess.run(self.update, feed_dict)
        return loss
    
    def predict(self, images):
        """Predict with given images
        
        Args:
            images (numpy array): Numpy array with (batch, small_height, small_width, time_batch), features of dataset
        
        Returns:
            list: List of 0 or 1, prediction for each image in images input
        """
        feed_dict = {self.model_in:images}
        return self.sess.run(self.model_out, feed_dict)
