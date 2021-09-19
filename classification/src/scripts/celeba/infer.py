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
from nets.simple_net import Classification_Net

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../conf", config_name="default")
def infer(cfg: DictConfig):
    ckpt_path = Path(cfg.datasets.celeba.base_url) / cfg.nets.myCNN.checkpoint_params.checkpoint_dir / \
                cfg.datasets.celeba.exp_name / cfg.validation_params.ckpt_file
    params = torch.load(ckpt_path)
    model = Classification_Net(conf=params["hyper_parameters"]["conf"],
                               nb_classes=params["hyper_parameters"]["nb_classes"])
    model.load_state_dict(params["state_dict"])

    img = Image.open(cfg.inference_params.img_path)
    img_tensor = transforms.ToTensor()(img)
    infer_transform = get_transforms(img_tensor.shape, mode="inference")
    img = infer_transform(img)
    prob, label = model.infer(img)
    prob, label = prob.detach().cpu().item(), label.detach().cpu().item()
    print("Label is", label, "with probability", prob)

if __name__ == "__main__":
    infer()