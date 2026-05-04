# Point-Query Quadtree for Crowd Counting, Localization, and More (ICCV 2023)

This repository includes the official implementation of the paper: 

[**Point-Query Quadtree for Crowd Counting, Localization, and More**](https://arxiv.org/abs/2308.13814)

International Conference on Computer Vision (ICCV), 2023

[Chengxin Liu](https://cxliu0.github.io/)<sup>1</sup>, [Hao Lu](https://sites.google.com/site/poppinace/)<sup>1</sup>, [Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>, [Tongliang Liu](https://tongliang-liu.github.io/)<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, China  

<sup>2</sup>The University of Sydney, Australia

[[Paper]](https://arxiv.org/abs/2308.13814) | [[Supplementary]](https://drive.google.com/file/d/1WxdtOaEEccYrXuNQTn1k29lFDAetBm63/view?usp=sharing)

![PET](teaser.JPG)

## Highlights

We formulate crowd counting as a decomposable point querying process, where sparse input points could split into four new points when necessary. This formulation exhibits many appealing properties:

- *Intuitive*: The input and output are both interpretable and steerable
  
- *Generic*: PET is applicable to a number of crowd-related tasks, by simply adjusting the input format
  
- *Effective*: PET reports state-of-the-art crowd counting and localization results
  

## Installation

- Required packages:
  
```
torch
torchvision
numpy
opencv-python
scipy
matplotlib
```

- Install packages:

```
pip install -r requirements.txt
```


## Data Preparation

- Download crowd-counting datasets, e.g., [ShanghaiTech](https://github.com/desenzhou/ShanghaiTechDataset).
  
- We expect the directory structure to be as follows:
  

```
PET
├── data
│    ├── ShanghaiTech
├── datasets
├── models
├── ...
```

- Alternatively, you can define the path of the dataset in [datasets/__init__.py](datasets/__init__.py)

- For [UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/), [JHU-Crowd++](http://www.crowd-counting.com/), and [NWPU-Crowd](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/) datasets, please refer to [preprocess_dataset.py](https://github.com/cxliu0/PET/blob/main/preprocess_dataset.py):

  * change [```dataset```](https://github.com/cxliu0/PET/blob/main/preprocess_dataset.py#L217) and [```data_root```](https://github.com/cxliu0/PET/blob/main/preprocess_dataset.py#L218)
  * run ```python preprocess_dataset.py```


## Training

- The default backbone is now `convnextv2_base` through a shared timm/FPN adapter in [models/backbones/backbone_timm.py](models/backbones/backbone_timm.py). Install `timm` from `requirements.txt`; pass `--no_pretrained_backbone` when running offline.
- The original `vgg16_bn` path is still available. For that path, download ImageNet pretrained [vgg16_bn](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth), and put it in ```pretrained``` folder. Or you can define your pre-trained model path in [models/backbones/vgg.py](models/backbones/vgg.py)

### Backbone choice for PET

For PET, `convnextv2_base` is the safest default starting point. It is not the only strong option, but it is the best balance of accuracy, stability, and integration effort for this codebase.

- ConvNeXt V2 is a pure CNN, so it fits PET's FPN-style feature flow cleanly.
- It gives hierarchical features with strong semantic quality, which matters for quadtree splitting and point localization.
- It is a much stronger upgrade than VGG16 without introducing the integration and deployment risk of more exotic architectures.

The timm adapter also supports these ablation families when they expose PET-compatible 4x and 8x feature maps:

- Dense accuracy candidates: `convnext_base`, `convnextv2_base`, `swinv2_base`, `maxvit_small`, `pvtv2_b1`.
- Latency candidates: `convnextv2_tiny`, `fastvit_tiny`, `efficientvit_tiny`, `mobilenetv4_small`, `repvit_tiny`, `edgenext_tiny`.
- Additional supported aliases: run `python main.py --list_backbones`.

I also considered the backbone families you listed:

- VMamba: very promising for high-resolution global context, but more experimental to integrate and tune here.
- InternImage: excellent for dense prediction and scale-aware localization, but usually heavier operationally.
- FastViT: best if latency is the main goal, but not my first pick when the priority is MAE reduction.
- Swin V2 / MaxViT: strong hybrid options and good secondary candidates if ConvNeXtV2 stalls.

Practical order to benchmark:

1. `convnextv2_base` as the baseline and tuning target.
2. `internimage` or `swinv2` if you want a higher-capacity dense predictor.
3. `fastvit` if deployment latency matters more than raw accuracy.
4. `vmamba` if you are willing to spend more time on tuning and backend validation.

### Negative images and quadtree training

- Empty annotation files and missing SHA `.mat` files are treated as valid zero-person images.
- Positive training images try multiple random crops (`--crop_attempts`) and keep the first crop with enough people (`--min_crop_points`) before falling back to the densest sampled crop.
- Classification and split-map losses are normalized so all-negative batches do not dominate positive localization.
- The quadtree splitter now has a ground-truth quality loss (`--quadtree_loss_coef`) built from local point counts, plus an adaptive split threshold (`--split_threshold -1`, `--split_threshold_quantile`).

### Ubuntu quick commands

```bash
git clone https://github.com/cxliu0/PET.git
cd PET
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# train ConvNeXtV2 on ShanghaiTech Part A
python -m torch.distributed.run --nproc_per_node=1 --master_port=10001 main.py \
  --dataset_file SHA \
  --data_path ./data/ShanghaiTech/part_A \
  --backbone convnextv2_base \
  --output_dir convnextv2_base \
  --epochs 1500 \
  --eval_freq 5

# quick backbone list
python main.py --list_backbones

# latency-oriented ablation dry run
python scripts/run_backbone_seeds.py \
  --preset latency \
  --seeds 42 7 \
  --extra_args "--dataset_file SHA --data_path ./data/ShanghaiTech/part_A --epochs 300 --eval_freq 5" \
  --dry_run
```
  

- To train PET on ShanghaiTech PartA, run
  
  ```
  sh train.sh
  ```
  

## Evaluation

- Modify [eval.sh](eval.sh)
  - change ```--resume``` to your local model path
- Run

```
sh eval.sh
```

## Pretrained Models

- Environment:
```
python==3.8
pytorch==1.12.1
torchvision==0.13.1
```

- Models:

| Dataset                  | Model Link  | Training Log  | MAE |
| ------------------------ | ----------- | --------------| ----|
| ShanghaiTech PartA       |  [SHA_model.pth](https://drive.google.com/file/d/1QwV8hrEDs1LQ4h1TH4KSL8tB51AImNMT/view?usp=drive_link)   | [SHA_log.txt](https://drive.google.com/file/d/1UpY61L0KWRA9c29CM9FMX34bHyprnPUY/view?usp=sharing) | 49.08 |
| ShanghaiTech PartB       |  [SHB_model.pth](https://drive.google.com/file/d/10HK42xC6fmOK-5lQfu-pTn6oAHYeRUhv/view?usp=sharing)   | [SHB_log.txt](https://drive.google.com/file/d/1M74PI0XuJtQraPOUiCQJSCUjrWoJUq3n/view?usp=sharing) | 6.18 |
| UCF_QNRF                 |  [UCF_QNRF.pth](https://drive.google.com/file/d/129l__gW51UtTQnPKM-90lTZo508-Eh7I/view?usp=sharing)    | - | - |
| JHU_Crowd                |  [JHU_Crowd.pth](https://drive.google.com/file/d/1D4vtoYhQuvj_5onJaXJRtWrlwrl2ckbE/view?usp=sharing)   | - | - |
| NWPU_Crowd               |  [NWPU_Crowd.pth](https://drive.google.com/file/d/1MX7tQAexyc9slrt7TaNSK7j8RtSvnI2H/view?usp=sharing)  | - | - |


## Frequently Asked Questions (FAQ)

* The model trained on my custom dataset does not perform well, why?
  * Please check the [load_data](https://github.com/cxliu0/PET/blob/main/datasets/SHA.py#L105) function in your custom dataset. The input format should be (y, x) instead of (x, y). If the input annotations are wrong during training, the output of the trained model could be abnormal.
 
* How to deal with images with no person?
  * Please refer to this [issue](https://github.com/cxliu0/PET/issues/33#issuecomment-2782560733).

## Citation

If you find this work helpful for your research, please consider citing:

```
@InProceedings{liu2023pet,
  title={Point-Query Quadtree for Crowd Counting, Localization, and More},
  author={Liu, Chengxin and Lu, Hao and Cao, Zhiguo and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```


## Permission

This code is for academic purposes only. Contact: Chengxin Liu (cx_liu@hust.edu.cn)


## Acknowledgement

We thank the authors of [DETR](https://github.com/facebookresearch/detr) and [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) for open-sourcing their work.


