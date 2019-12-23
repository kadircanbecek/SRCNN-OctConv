import os

from tensorflow_core.python.keras.callbacks import ModelCheckpoint, LambdaCallback

from custom_callbacks import BicubicPSNR
from custom_generator import SRCNNGenerator
from oct_conv.octConvClass import OctConv
from utils import LoadWeight, get_filenames, psnr


def create_srcnn(name='SRCNN', metrics=None):
    if metrics is None:
        metrics = ['accuracy', psnr]
    octconv = OctConv()
    octconv.add_initial_layer(filters=128, kernel_size=(9, 9))
    octconv.add_oct_conv_block(filters=64, kernel_size=(1, 1))
    octconv.add_final_oct_conv_layer(filters=128, kernel_size=(5, 5))
    octconv.construct_model(
        name=name,
        metrics=metrics
    )
    return octconv.get_model()


# Properties
data_path = "DATASET/BSDS500"
batch_size = 10

srcnn_model = create_srcnn()
# Dataset filepath
data_raw_lr_filenames = get_filenames(data_path + '_BICUBIC')
data_raw_hr_filenames = get_filenames(data_path + '_CROPPED')
batch_generator_train = SRCNNGenerator(data_raw_lr_filenames, data_raw_hr_filenames, batch_size=batch_size)

# calculates bicubic psnr before training prints after every epoch
bicubic_callback = BicubicPSNR(train_generator=batch_generator_train)

# Checkpoint part
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Initialize saver
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=1)

# Return model and initial_epoch
srcnn_model, initial_epoch = LoadWeight(
    filepath=checkpoint_dir,
    model=srcnn_model
).load_weight()

# callback for shuffling filenames
shuffle_callback = LambdaCallback(on_epoch_end=lambda epoch, logs=None: batch_generator_train.shuffle_names())

# Fit function
srcnn_model.fit_generator(generator=batch_generator_train,
                          # steps_per_epoch=int(len(data_raw_hr_filenames) / batch_size),
                          epochs=10,
                          initial_epoch=initial_epoch,
                          verbose=1,
                          steps_per_epoch=1,
                          max_queue_size=10,
                          shuffle=False,
                          workers=10,
                          callbacks=[bicubic_callback, cp_callback, shuffle_callback])
