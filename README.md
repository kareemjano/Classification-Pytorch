# Classification-Pytorch

This project trains a BCInception CNN, pretrained on ImageNet, on the classification task using Celeba dataset.

The main purpose for this project is to show how to integrate some of the most advanced ML tools to build a complete ML project. The project makes use of the following tools/libraries: [Pytorch Lightning](https://www.pytorchlightning.ai/), [DVC](https://dvc.org/), [Hydra](https://hydra.cc/), [MLFlow](https://mlflow.org/), [imgaug](https://imgaug.readthedocs.io/en/latest/), [Tensorboard](https://www.tensorflow.org/tensorboard), [Docker](https://www.docker.com/).

## Running the project:

### Presets

Set the correct configurations  in classification/conf/default.yaml file.

Initialize DVC inside the classification directory by running `dvc init --subdir`

(Optional) To track the data, add dvc remote by following the instruction mentioned [here](https://dvc.org/doc/command-reference/remote/add).

(Optional) Set the preferred caching behavior. Ex, `dvc config cache.type hardlink,symlink`. More information can be found [here](https://dvc.org/doc/user-guide/large-dataset-optimization)

### Running inside a Docker container

In the projects root directory run the following commands: 

```
docker-compose build
docker-compose up
```

When done run: `docker-compose down`

### Running without a Docker container

Run `dvc repro` or `dvc repro --no-commit` if tracking the data is not required.

