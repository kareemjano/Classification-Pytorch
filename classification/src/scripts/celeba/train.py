import hydra
from omegaconf import DictConfig
import torch
from torchsummary import summary
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
import logging
import sys
sys.path.append('src')

from datasets.celeba.classification_dataloader import Classification_Dataloader
from nets.simple_net import Classification_Net

logger = logging.getLogger(__name__)
def get_callbacks(cfg):
    early_stop_callback = pl.callbacks.EarlyStopping(
        **cfg.nets.myCNN.early_stopping_params
    )

    checkpoint_params = cfg.nets.myCNN.checkpoint_params
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=checkpoint_params.monitor,
        dirpath=str(Path(cfg.datasets.celeba.base_url) /
                    checkpoint_params.checkpoint_dir /
                    cfg.datasets.celeba.exp_name),
        filename=checkpoint_params.filename,
        save_top_k=checkpoint_params.save_top_k,
        mode=checkpoint_params.mode,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()

    return [
        early_stop_callback,
        checkpoint_callback,
        lr_monitor,
    ]


@hydra.main(config_path="../../../conf", config_name="default")
def train(cfg: DictConfig):
    celeba = Classification_Dataloader(conf=cfg, batch_size=16, num_workers=12, image_aug_p=0)
    celeba.setup()
    model = Classification_Net(cfg, nb_classes=celeba.nb_classes())

    if torch.cuda.is_available():
        model.to('cuda')

    logger.info('model summary')
    logger.info(summary(model, [(3, cfg.nets.myCNN.params.input_shape[1], cfg.nets.myCNN.params.input_shape[2])]))
    model.cpu()

    callbacks = get_callbacks(cfg)

    stats_logger = None
    if cfg.logger_params.logger.lower() == "tensorboard":
        stats_logger = TensorBoardLogger(
            Path(cfg.datasets.celeba.base_url) / cfg.logger_params.logger_dir,
            name=cfg.datasets.celeba.exp_name
        )
    elif cfg.logger_params.logger.lower() == "mlflow":
        stats_logger = MLFlowLogger(
            experiment_name=cfg.datasets.celeba.exp_name,
            tracking_uri=str(Path(cfg.datasets.celeba.base_url) / cfg.logger_params.logger_dir)
        )
    else:
        logger.info("No valid logger is specified.")

    trainer = pl.Trainer(**cfg.nets.myCNN.trainer_params, logger=stats_logger, callbacks=callbacks)
    trainer.fit(model, celeba)

    if cfg.validation_params.run_val:
        trainer.test(model, celeba.val_dataloader(), ckpt_path='best')
    if cfg.validation_params.run_test:
        trainer.test(model, celeba.test_dataloader(), ckpt_path='best')


if __name__ == "__main__":
    train()