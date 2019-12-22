import numpy as np
from cv2 import imread
from tensorflow import keras

import utils


class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.label_names = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        image_x_raw = utils.read_data(batch_x)
        batch_y = self.label_names[idx * self.batch_size: (idx + 1) * self.batch_size]
        image_y_raw = utils.read_data(batch_y)
        image_x = []
        image_y = []
        for i in range(len(batch_x)):
            image_x_tmp, image_y_tmp = utils.preprocessing(image_x_raw[i]), utils.preprocessing(image_y_raw[i])
            image_x.extend(image_x_tmp)
            image_y.extend(image_y_tmp)
        image_x_np, image_y_np = np.array(image_x), np.array(image_y)
        return image_x_np, image_y_np
