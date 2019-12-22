from tensorflow_core.python.keras import Input, Model

import utils
from customGenerator import My_Custom_Generator
from octConvLayers import OctConvInitialLayer, OctConvBlockLayer, OctConvFinalLayer

ip = Input(shape=(32, 32, 3))
x_low, x_high = OctConvInitialLayer(128, kernel_size=(9, 9))(ip)
x_low, x_high = OctConvBlockLayer(64, kernel_size=(1, 1))([x_low, x_high])
x = OctConvFinalLayer(3, kernel_size=(5, 5))([x_low, x_high])
model = Model(ip, x, name="SRCNN")
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy", utils.bicubic_psnr(ip), utils.psnr])
data_path = "DATASET/BSDS500"
data_raw_lr_filenames = utils.get_filenames(data_path + '_BICUBIC')
data_raw_hr_filenames = utils.get_filenames(data_path + '_CROPPED')
# data_lr = []
# data_hr = []
# for i in range(len(data_raw_lr)):
#     data_lr_tmp, data_hr_tmp = utils.preprocessing(data_raw_lr[i]), utils.preprocessing(data_raw_hr[i])
#     data_lr.extend(data_lr_tmp)
#     data_hr.extend(data_hr_tmp)

model.summary()
batch_size = 10
batch_generator_train = My_Custom_Generator(data_raw_lr_filenames, data_raw_hr_filenames, batch_size=batch_size)
model.fit_generator(generator=batch_generator_train,
                    steps_per_epoch=int(len(data_raw_hr_filenames) / batch_size),
                    epochs=10,
                    verbose=1,
                    use_multiprocessing=True,
                    workers=10)
