from random import shuffle

import numpy as np
from tensorflow import keras

import utils


class SRCNNGenerator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.label_names = labels
        self.batch_size = batch_size

    def shuffle_names(self):
        data_shuffle = list(zip(self.image_filenames, self.label_names))
        shuffle(data_shuffle)
        self.image_filenames, self.label_names = zip(*data_shuffle)
        print("names shuffled")

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        # print('idx ', idx)
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
        image_x_np, image_y_np = np.array(image_x, dtype=np.float32), np.array(image_y, dtype=np.float32)
        return image_x_np, image_y_np

    def get_batch_amount_per_epoch(self):
        return len(self.image_filenames) // self.batch_size
    # def on_epoch_end
