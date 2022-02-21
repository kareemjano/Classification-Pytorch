import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys
sys.path.append('src')

from image_toolbox.dataset_utils.celeba_utils import celeba_save_images_in_folders
from image_toolbox.dataset_utils.file_utils import unzip_file

def check_cfg(cfg: DictConfig, urls):
    assert not cfg.datasets is None, "datasets could not be found in cfg"
    assert not cfg.datasets.celeba is None, "datasets.celeba could not be found in cfg"

    for key, value in list(urls.items()):
        if 'url' in key:
            p = Path(value)
            assert p.exists(), str(p) + " doesnt exist. " + "key "+key


@hydra.main(config_path="../../../conf", config_name="default")
def to_folders(cfg: DictConfig) -> None:
    import os

    print("Preparing Celeba...")
    print("cw dir", os.getcwd())
    print(os.listdir(cfg.datasets.celeba.base_url))
    print(cfg.datasets.celeba.base_url)

    urls = {
        "base_url": Path(cfg.datasets.celeba.base_url).resolve()
    }
    print(urls["base_url"].resolve())
    print(cfg.datasets.celeba.zip_file_name in os.listdir(os.path.join(urls["base_url"])))
    print(os.path.join(urls["base_url"], cfg.datasets.celeba.zip_file_name))
    print(os.path.exists(os.path.join(urls["base_url"], cfg.datasets.celeba.zip_file_name)))
    urls.update({
        "zip_file_url": urls['base_url'] / cfg.datasets.celeba.zip_file_name,
        "labels_txt_url": urls['base_url'] / cfg.datasets.celeba.labels_txt_file,
        "split_txt_url": urls['base_url'] / cfg.datasets.celeba.split_txt_file,
    })


    check_cfg(cfg, urls)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    print(urls["zip_file_url"])
    imgs_path = unzip_file(urls["zip_file_url"], urls['base_url'])

    celeba_save_images_in_folders(imgs_path, urls["labels_txt_url"])


if __name__ == "__main__":
    to_folders()
