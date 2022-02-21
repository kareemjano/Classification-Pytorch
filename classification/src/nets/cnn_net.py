import torch
from torch import nn
from .general_nets import FCN_layer
from .my_CNNs import my_CNN
from torch.optim.lr_scheduler import StepLR
import pretrainedmodels
from image_toolbox.custom_lightning_modules import ClassificationModule


class Classification_Net(ClassificationModule):
    """
    Simple Classification Model
    """

    def __init__(self, cfg):
        """
        configuration parameters containing params and hparams.
        """
        super().__init__(cfg)
        self.save_hyperparameters()
        self.scheduler_params = self.net_params.scheduler_params

        if self.model_select == "bcinception":
            self.model = pretrainedmodels.bninception(num_classes=1000, pretrained="imagenet")
            self.model.last_linear = nn.Linear(1024, self.dataset_cfg.n_classes)
        elif self.model_select == "simple_cnn":
            channels, height, width = self.input_shape
            self.cnn_output_size = int(self.hparam["filter_channels"] * 4 * int(height / 8) * int(width / 8))
            self.model = nn.Sequential(
                my_CNN(nn.PReLU, self.hparam["filter_size"], self.hparam["filter_channels"],
                       padding=int(self.hparam["filter_size"] / 2), input_shape=self.input_shape
                       ),
                nn.Flatten(),
                FCN_layer(self.cnn_output_size, self.hparam['n_hidden1']),
                FCN_layer(self.hparam['n_hidden1'], self.hparam['n_hidden2'], dropout=self.hparam['dropout']),
                nn.Linear(self.hparam['n_hidden2'], self.dataset_cfg.n_classes),
            )


    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.hparam["lr"],
                                     weight_decay=self.hparam["weight_decay"])

        if self.scheduler_params is not None:
            lr_scheduler = StepLR(optimizer, **self.scheduler_params)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer