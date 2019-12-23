from tensorflow.keras.layers import Conv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


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
        """
            Initializes the Octave Convolution architecture.
            Accepts a single input tensor, and returns a pair of tensors.
            The first tensor is the high frequency pathway.
            The second tensor is the low frequency pathway.
            # Arguments:
                ip: keras tensor.
                filters: number of filters in conv layer.
                kernel_size: conv kernel size.
                strides: strides of the conv.
                alpha: float between [0, 1]. Defines the ratio of filters
                    allocated to the high frequency and low frequency
                    branches of the octave conv.
                padding: padding mode.
                dilation: dilation conv kernel.
                bias: bool, whether to use bias or not.
            # Returns:
                a pair of tensors:
                    - x_high: high frequency pathway.
                    - x_low: low frequency pathway.
            """
        if dilation is None:
            dilation = (1, 1)

        high_low_filters = int(alpha * filters)
        high_high_filters = filters - high_low_filters

        if strides[0] > 1:
            ip = AveragePooling2D()(self.ip)

        # High path
        self.x_high = Conv2D(high_high_filters, kernel_size, padding=padding,
                             dilation_rate=dilation, use_bias=bias,
                             kernel_initializer='he_normal', activation=self.activation)(self.ip)

        # Low path
        x_high_low = AveragePooling2D()(self.ip)
        self.x_low = Conv2D(high_low_filters, kernel_size, padding=padding,
                            dilation_rate=dilation, use_bias=bias,
                            kernel_initializer='he_normal', activation=self.activation)(x_high_low)

    def add_oct_conv_block(self, filters, kernel_size=(3, 3), strides=(1, 1),
                           alpha=0.5, padding='same', dilation=None, bias=False):
        """
        Constructs an Octave Convolution block.
        Accepts a pair of input tensors, and returns a pair of tensors.
        The first tensor is the high frequency pathway for both ip/op.
        The second tensor is the low frequency pathway for both ip/op.
        # Arguments:
            self.x_high: keras tensor.
            self.x_low: keras tensor.
            filters: number of filters in conv layer.
            kernel_size: conv kernel size.
            strides: strides of the conv.
            alpha: float between [0, 1]. Defines the ratio of filters
                allocated to the high frequency and low frequency
                branches of the octave conv.
            padding: padding mode.
            dilation: dilation conv kernel.
            bias: bool, whether to use bias or not.
        # Returns:
            a pair of tensors:
                - x_high: high frequency pathway.
                - x_low: low frequency pathway.
        """
        if dilation is None:
            dilation = (1, 1)

        low_low_filters = high_low_filters = int(alpha * filters)
        high_high_filters = low_high_filters = filters - low_low_filters

        avg_pool = AveragePooling2D()

        if strides[0] > 1:
            self.x_high = avg_pool(self.x_high)
            self.x_low = avg_pool(self.x_low)

        # High path
        x_high_high = Conv2D(high_high_filters, kernel_size, padding=padding,
                             dilation_rate=dilation, use_bias=bias,
                             kernel_initializer='he_normal', activation=self.activation)(self.x_high)

        x_low_high = Conv2D(low_high_filters, kernel_size, padding=padding,
                            dilation_rate=dilation, use_bias=bias,
                            kernel_initializer='he_normal', activation=self.activation)(self.x_low)
        x_low_high = UpSampling2D(interpolation='nearest')(x_low_high)

        # Low path
        x_low_low = Conv2D(low_low_filters, kernel_size, padding=padding,
                           dilation_rate=dilation, use_bias=bias,
                           kernel_initializer='he_normal', activation=self.activation)(self.x_low)

        x_high_low = avg_pool(self.x_high)
        x_high_low = Conv2D(high_low_filters, kernel_size, padding=padding,
                            dilation_rate=dilation, use_bias=bias,
                            kernel_initializer='he_normal', activation=self.activation)(x_high_low)

        # Merge paths
        self.x_high = add([x_high_high, x_low_high])
        self.x_low = add([x_low_low, x_high_low])

    def add_final_oct_conv_layer(self, filters, kernel_size=(3, 3), strides=(1, 1),
                                 padding='same', dilation=None, bias=False):
        """
        Ends the Octave Convolution architecture.
        Accepts two input tensors, and returns a single output tensor.
        The first input tensor is the high frequency pathway.
        The second input tensor is the low frequency pathway.
        # Arguments:
            self.x_high: keras tensor.
            self.x_low: keras tensor.
            filters: number of filters in conv layer.
            kernel_size: conv kernel size.
            strides: strides of the conv.
            padding: padding mode.
            dilation: dilation conv kernel.
            bias: bool, whether to use bias or not.
        # Returns:
            a single Keras tensor:
                - x_high: The merged high frequency pathway.
        """
        if dilation is None:
            dilation = (1, 1)

        if strides[0] > 1:
            avg_pool = AveragePooling2D()

            self.x_high = avg_pool(self.x_high)
            self.x_low = avg_pool(self.x_low)

        # High path
        x_high_high = Conv2D(filters, kernel_size, padding=padding,
                             dilation_rate=dilation, use_bias=bias,
                             kernel_initializer='he_normal', activation=self.activation)(self.x_high)

        # Low path
        x_low_high = Conv2D(filters, kernel_size, padding=padding,
                            dilation_rate=dilation, use_bias=bias,
                            kernel_initializer='he_normal', activation=self.activation)(self.x_low)

        x_low_high = UpSampling2D(interpolation='nearest')(x_low_high)

        # Merge paths
        self.x = add([x_high_high, x_low_high])

    def construct_model(self, ):
        self.model = Model(self.ip, self.x)
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)
        self.model.summary()

    # def train_batch(self):


if __name__ == "__main__":
    octconv = OctConv()
    octconv.add_initial_layer(128, (9, 9))
    octconv.add_oct_conv_block(64, (1, 1))
    octconv.add_final_oct_conv_layer(3, (5, 5))
    octconv.construct_model()
