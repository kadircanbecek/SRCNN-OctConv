from tensorboard.plugins.hparams.keras import Callback

from utils import mean_squared_error, psnr_for_loss


class BicubicPSNR(Callback):
    def __init__(self, train_generator):
        super(Callback, self).__init__()

        self.train_generator = train_generator
        self.bicubic_loss = 0

    def on_train_begin(self, logs=None):
        batch_amount = self.train_generator.get_batch_amount_per_epoch()
        for batch_number in range(batch_amount):
            features, y_true = self.train_generator.__getitem__(batch_number)
            bicubic_loss_batch = mean_squared_error(features, y_true)
            self.bicubic_loss += bicubic_loss_batch / self.train_generator.get_batch_amount_per_epoch()
        print("Bicubic LOSS: {0}".format(self.bicubic_loss))
        print("Bicubic PSNR: {0}".format(psnr_for_loss(self.bicubic_loss)))
        # psnr_value = psnr(features, y_true)
        # print('PSNR for bicubic is {0}'.format(psnr_value))

    def on_epoch_end(self, epoch, logs=None):
        print("Bicubic PSNR: {0}".format(psnr_for_loss(self.bicubic_loss)))