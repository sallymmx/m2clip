# [AAAI 2024 Oral] M2CLIP: A Multimodal, Multi-Task Adapting Framework for Video Action Recognition

This is the official repo of the paper [M2CLIP: A Multimodal, Multi-Task Adapting Framework for Video Action Recognition](https://ojs.aaai.org/index.php/AAAI/article/download/28361/28707).

```
@inproceedings{wang2024multimodal,
  title={A Multimodal, Multi-Task Adapting Framework for Video Action Recognition},
  author={Wang, Mengmeng and Xing, Jiazheng and Jiang, Boyuan and Chen, Jun and Mei, Jianbiao and Zuo, Xingxing and Dai, Guang and Wang, Jingdong and Liu, Yong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5517--5525},
  year={2024}
}

```

## Environment

We use conda to manage the Python environment. The dumped configuration is provided at [environment.yml](environment.yml)

## Configuration

Some common configurations (e.g., dataset paths, pretrained backbone paths) are set in `config.py`. We've included an example configuration in `config.py.example` which contains all required fields with values left empty. Please copy `config.py.example` to `config.py` and fill in the values before running the models.

## Dataset preparation

The data list should be organized as follows

```
<video_1> <label_1>
<video_2> <label_2>
...
<video_N> <label_N>
```

where `<video_i>` is the path to a video file, and `<label_i>` is an integer between $0$ and $M-1$ representing the class of the $i$-th video, where $M$ is the total number of classes.

We release the data list we used for Kinetics-400 (`k400`, [train list link](https://drive.google.com/file/d/1RbuTI5foZTrPaCTsAsrj99bKgUWrFiPZ/view?usp=sharing), [val list link](https://drive.google.com/file/d/1quRzJYZslobQb-fwTV7gDLgzk9Pu9fXl/view?usp=sharing)) and Something-something-v2 (`ssv2`, [train list link](https://drive.google.com/file/d/10ZGWG5WsPxl6-56xO8_e2TacD6SFzPEC/view?usp=sharing), [val list link](https://drive.google.com/file/d/1i9ED1vU-yoYK5L89_X1J1fwsq-XyHX7B/view?usp=sharing)), which reflect the class mapping of the released models and the videos available at our side. It is strongly recommended that the Kinetics-400 lists be cleaned first, as some videos may have been taken down by YouTube for various reasons (the training will stop on broken videos in the current implementation).

After obtaining the videos and the data lists, set the root dir and the list paths in `config.py` in the `DATASETS` dictionary (fill in the blanks for `k400` and `ssv2` or add new items for custom datasets). For each dataset, 5 fields are required:

* `TRAIN_ROOT`: the root directory which the video paths in the training list are relative to.

* `VAL_ROOT`: the root directory which the video paths in the validation list are relative to.

* `TRAIN_LIST`: the path to the training video list.

* `VAL_LIST`: the path to the validation video list.

* `NUM_CLASSES`: number of classes of the dataset.

## Backbone preparation

We use the CLIP checkpoints from the [official release](https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L30). Put the downloaded checkpoint paths in `config.py`. The currently supported architectures are CLIP-ViT-B/16 (set `CLIP_VIT_B16_PATH`) and CLIP-ViT-L/14 (set `CLIP_VIT_B16_PATH`). 

## Run the models

We provide some preset scripts in the [scripts/](scripts/) directory containing some recommended settings. For a detailed description of the comand line arguments see the help message of `main.py`.



The CLIP model implementation is modified from [CLIP official repo](https://github.com/openai/CLIP). This repo is built upon [ST-Adapter](https://github.com/linziyi96/st-adapter). Thanks for their awesome works!
