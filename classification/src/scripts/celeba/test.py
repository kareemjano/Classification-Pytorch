import hydra
from omegaconf import DictConfig
from pathlib import Path
import pytorch_lightning as pl
import logging
import sys
sys.path.append('src')

from datasets.celeba.classification_dataloader import Classification_Dataloader
from nets.cnn_net import Classification_Net

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../conf", config_name="default")
def test(cfg: DictConfig):
    celeba = Classification_Dataloader(conf=cfg, batch_size=16, num_workers=12, image_aug_p=0)
    celeba.setup()
    model_select = cfg.nets.select
    net_params = cfg.nets[model_select]

    ckpt_path = Path(cfg.dataset.base_url) / net_params.checkpoint_params.checkpoint_dir / \
                net_params.exp_name / net_params.validation_params.ckpt_file

    model = Classification_Net.load_from_checkpoint(ckpt_path)
    trainer = pl.Trainer(**net_params.trainer_params)
    trainer.test(model, celeba.test_dataloader())


if __name__ == "__main__":
    test()