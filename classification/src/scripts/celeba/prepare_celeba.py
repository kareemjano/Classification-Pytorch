import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys
sys.path.append('src')

from image_toolbox.dataset_utils.celeba_utils import celeba_save_images_in_folders, celeba_split_from_file
from image_toolbox.dataset_utils.file_utils import unzip_file

def check_cfg(cfg: DictConfig, urls):
    assert not cfg.datasets is None, "datasets could not be found in cfg"
    assert not cfg.datasets.celeba is None, "datasets.celeba could not be found in cfg"

    for key, value in list(urls.items()):
        if 'url' in key:
            p = Path(value)
            assert p.exists(), str(p.resolve()) + " doesnt exist. " + "key "+key


@hydra.main(config_path="../../../conf", config_name="default")
def to_folders(cfg: DictConfig) -> None:
    urls = {
        "base_url": Path(cfg.datasets.celeba.base_url)
    }

    urls.update({
        "data_url": urls['base_url'] / cfg.datasets.celeba.data_dir_name,
        "zip_file_url": urls['base_url'] / cfg.datasets.celeba.zip_file_name,
        "labels_txt_url": urls['base_url'] / cfg.datasets.celeba.labels_txt_file,
        "split_txt_url": urls['base_url'] / cfg.datasets.celeba.split_txt_file,
    })


    if not urls["data_url"].exists():
        urls["data_url"].mkdir(parents=True)

    check_cfg(cfg, urls)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    imgs_path = unzip_file(urls["zip_file_url"], urls['data_url'])

    map = None
    if urls["split_txt_url"] is not None:
        map = celeba_split_from_file(imgs_path, urls["split_txt_url"])

    celeba_save_images_in_folders(imgs_path, urls["labels_txt_url"], map=map)


if __name__ == "__main__":
    to_folders()
