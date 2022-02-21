import hydra
from omegaconf import DictConfig
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import logging
import sys
sys.path.append('src')

from datasets.celeba.tranforms import get_transforms
from nets.cnn_net import Classification_Net

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../conf", config_name="default")
def infer(cfg: DictConfig):
    model_select = cfg.nets.select
    net_params = cfg.nets[model_select]

    ckpt_path = Path(cfg.dataset.base_url) / net_params.checkpoint_params.checkpoint_dir / \
                net_params.exp_name / net_params.inference_params.ckpt_file
    model = Classification_Net.load_from_checkpoint(ckpt_path)

    img = Image.open(net_params.inference_params.img_path)
    infer_transform = get_transforms(net_params.params.input_shape, mode="inference")
    img = infer_transform(img)
    prob, label = model.infer(img)
    prob, label = prob.detach().cpu().item(), label.detach().cpu().item()
    print(f"Label is {label} with probability {prob*100:2.2f}")

if __name__ == "__main__":
    infer()