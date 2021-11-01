import hydra
import os
import sys
sys.path.append('src')
from image_toolbox.dataset_utils.file_utils import download_file_from_google_drive

file_list = [
        # File ID                         Filename
        ("19WCblcbXjfqAQRmA7x0OtOAs6rMVvrnn", "img_align_celeba.zip"),
        ("1jeEflo6J1siudrzKJGVMsJvZGoc6Pj_g", "identity_CelebA.txt"),
        ("1z5TT26C6-e1LAQ-7Bbv1MMQVnFpHW6YN", "list_eval_partition.txt"),
    ]

@hydra.main(config_path="../../../conf", config_name="default")
def download_celeba_dataset(cfg):
    for (file_id, filename) in file_list:
        output_path = os.path.join(cfg.datasets.celeba.base_url, filename)
        if not os.path.exists(output_path):
            download_file_from_google_drive(output_path, id=file_id)
        else:
            print(f"{filename} already exixts. Skipped!!!")

if __name__ == "__main__":
    download_celeba_dataset()
