import torch
from torch import nn
import pytorch_lightning as pl
from .general_nets import FCN_layer
from .my_CNNs import my_CNN
from torch.optim.lr_scheduler import StepLR
import pretrainedmodels


class Classification_Net(pl.LightningModule):
    """
    Simple Classification Model
    """

    def __init__(self, conf, nb_classes):
        """
        configuration parameters containing params and hparams.
        """
        super().__init__()
        self.save_hyperparameters()

        self.conf = conf
        self.model_select = conf.nets.select
        self.net_params = conf.nets[self.model_select]
        self.model_params = self.net_params.params
        self.hparam = self.net_params.hparams
        self.scheduler_params = self.net_params.scheduler_params
        # CNN
        self.input_shape = tuple(self.model_params["input_shape"])

        if self.model_select == "bcinception":
            self.conv = pretrainedmodels.bninception(num_classes=1000, pretrained="imagenet")
            self.conv.last_linear = nn.Linear(1024, nb_classes)
        elif self.model_select == "simple_cnn":
            self.conv = my_CNN(nn.PReLU,
                               self.hparam["filter_size"],
                               self.hparam["filter_channels"],
                               padding=int(self.hparam["filter_size"] / 2),
                               input_shape=self.input_shape
                               )  # size/8 x 4*channels

            channels, height, width = self.input_shape
            self.cnn_output_size = int(self.hparam["filter_channels"] * 4 * int(height / 8) * int(width / 8))

            self.out = nn.Sequential(
                FCN_layer(self.cnn_output_size, self.hparam['n_hidden1']),
                FCN_layer(self.hparam['n_hidden1'], self.hparam['n_hidden2'], dropout=self.hparam['dropout']),
                nn.Linear(self.hparam['n_hidden2'], nb_classes),
            )

        self.params_to_update = []
        self.params_to_update = self.parameters()

    def forward(self, x):
        x = self.conv(x)
        if self.model_select == "simple_cnn":
            x = x.view(x.size()[0], -1)
            if len(x.shape) == 3:
                x = x.squeeze()
            x = self.out(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.hparam["lr"],
                                     weight_decay=self.hparam["weight_decay"])

        if self.scheduler_params is not None:
            lr_scheduler = StepLR(optimizer, **self.scheduler_params)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def calc_acc(self, y, logits):
        y_hat = nn.Softmax(dim=1)(logits)
        y_hat = torch.argmax(y_hat, dim=1)
        assert y_hat.shape == y.shape, "shape of prediction doesnt match ground truth labels"

        return (y_hat == y).sum() / y.size(0)

    def general_step(self, batch):
        x, y = batch
        output = self(x)
        loss = nn.CrossEntropyLoss()(output, y)
        n_correct = self.calc_acc(y, output)

        return loss, n_correct

    def training_step(self, train_batch, batch_idx):
        self.train()
        loss, n_correct = self.general_step(train_batch)
        return {
            'loss': loss,
            'n_correct': n_correct,
        }

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        loss, n_correct = self.general_step(val_batch)
        return {
            'loss': loss.detach().cpu(),
            'n_correct': n_correct.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def general_epoch_end(self, outputs, mode):  ### checked
        # average over all batches aggregated during one epoch
        logger = False if mode == 'test' else True
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['n_correct'] for x in outputs]).mean()
        self.log(f'{mode}_loss', avg_loss, logger=logger)
        self.log(f'{mode}_acc', avg_acc, logger=logger)
        return avg_loss, avg_acc

    def training_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        avg_loss, avg_acc = self.general_epoch_end(outputs, 'test')
        return {
            'avg_loss': avg_loss,
            'avg_acc': avg_acc
        }

    def infer(self, images: torch.Tensor):
        self.eval()
        if len(images.size()) == 3:
            images = images.unsqueeze(0)

        output = self(images)
        output = nn.Softmax(dim=1)(output)
        probs, indices = torch.max(output, dim=1)
        return probs, indices
