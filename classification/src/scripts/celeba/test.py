import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import pytorch_lightning as pl
import logging
import sys
sys.path.append('src')

from datasets.celeba.classification_dataloader import Classification_Dataloader
from nets.simple_net import Classification_Net

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../conf", config_name="default")
def test(cfg: DictConfig):
    celeba = Classification_Dataloader(conf=cfg, batch_size=16, num_workers=12, image_aug_p=0)
    celeba.setup()

    ckpt_path = Path(cfg.datasets.celeba.base_url) / cfg.nets.myCNN.checkpoint_params.checkpoint_dir / \
                cfg.datasets.celeba.exp_name / cfg.validation_params.ckpt_file
    params = torch.load(ckpt_path)
    model = Classification_Net(conf=params["hyper_parameters"]["conf"],
                               nb_classes=params["hyper_parameters"]["nb_classes"])

    model.load_state_dict(params["state_dict"])

    trainer = pl.Trainer(**cfg.nets.myCNN.trainer_params, resume_from_checkpoint=ckpt_path)
    trainer.test(model, celeba.test_dataloader())


if __name__ == "__main__":
    test()