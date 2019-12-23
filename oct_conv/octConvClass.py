from tensorflow import shape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from oct_conv.octConvLayers import OctConvInitialLayer, OctConvBlockLayer, OctConvFinalLayer


class OctConv():
    def __init__(self, input_size=(32, 32, 3), optimizer="adam", activation="relu", loss="mean_squared_error"):
        self.ip = Input(shape=input_size)
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.x_low = None
        self.x_high = None
        self.x = None
        self.model = None

    def add_initial_layer(self, filters, kernel_size=(3, 3), strides=(1, 1),
                          alpha=0.5, padding='same', dilation=None, bias=False):
        self.x_low, self.x_high = OctConvInitialLayer(filters, kernel_size=kernel_size, strides=strides,
                                                      alpha=alpha, padding=padding, dilation=dilation, bias=bias,
                                                      activation=self.activation)(self.ip)

    def add_oct_conv_block(self, filters, kernel_size=(3, 3), strides=(1, 1),
                           alpha=0.5, padding='same', dilation=None, bias=False):
        self.x_low, self.x_high = OctConvBlockLayer(filters, kernel_size=kernel_size, strides=strides,
                                                    alpha=alpha, padding=padding, dilation=dilation, bias=bias,
                                                    activation=self.activation)([self.x_low, self.x_high])

    def add_final_oct_conv_layer(self, filters=None, kernel_size=(3, 3), strides=(1, 1),
                                 padding='same', dilation=None, bias=False):
        if filters is None:
            filters = self.ip.get_shape()[-1]
        self.x = OctConvFinalLayer(filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   dilation=dilation, bias=bias, activation=self.activation)([self.x_low, self.x_high])

    def construct_model(self, name='OctConv', metrics=None):
        if metrics is None:
            metrics = ["accuracy"]
        self.model = Model(self.ip, self.x, name=name)
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=metrics)
        self.model.summary()

    def get_model(self):
        return self.model

    # def train_batch(self):


if __name__ == "__main__":
    octconv = OctConv()
    octconv.add_initial_layer(128, (9, 9))
    octconv.add_oct_conv_block(64, (1, 1))
    octconv.add_final_oct_conv_layer(kernel_size=(5, 5))
    octconv.construct_model()
