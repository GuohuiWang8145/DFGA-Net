## DFGA-Net
DFGA-Net: Achieving Efficient and High-Accuracy Stereo Matching through Dual-Feature Extraction and Lightweight Gated Attention

## Introduction

Achieving a harmonious balance between efficiency and precision in stereo matching remains a daunting challenge, especially in time-sensitive domains such as autonomous driving. DFGA-Net proposes an efficient and accurate stereo matching approach through a novel Dual-Features Gated Attention Network. It employs a Dual-Features Extractor that integrates Instance Normalization and Batch Normalization to capture both personalized and generalized sample features, addressing environmental variability. Furthermore, DFGA-Net utilizes a Lightweight Transformer architecture with Lightweight Gated Attention Units to efficiently capture long-range dependencies, enhancing feature richness. Experimental results reveal that DFGA-Net attains exceptional efficiency and performance across a variety of datasets, including SceneFlow, KITTI 2012, KITTI 2015, and ETH3D, outperforming accuracy-centric models.
![main](./figs/framework.png)

## Performance
![kitti2015](./figs/kitti2015.png)

## Installation

run command line as following

```shell
   conda env create -f environment.yml
   conda activate DFGANet
```

## Datasets
Download [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), [ETH3D](https://www.eth3d.net/)

By default the dataloader [datasets.py](dataloader/datasets.py) assumes the datasets are located in folder `datasets` and are organized as follows:

```
datasets=
├── SceneFlow
│   ├── FlyingThings3D
│   │   └── frames_finalpass
│   ├── Monkaa
│   │   └── frames_finalpass
│   ├── Driving
│   │   └── frames_finalpass
├── KITTI
│   ├── kitti2012
│   │   ├── testing
│   │   └── training
│   ├── kitti2015
│   │   ├── testing
│   │   └── training
├── ETH3D
│   ├── test
│   └── train
```


## Evaluation

The evaluation scripts utilized to replicate the figures presented in our paper are provided in [eval.sh](eval.sh)

```shell
   sh eval.sh
```

For submissions to the KITTI and ETH3D online test sets, you may execute [submission.sh](submission.sh). The results can be submitted directly.

```shell
   sh submission.sh
```

## Training

All training scripts for different model variants on different datasets can be found in [train.sh](train.sh).

```shell
   sh train.sh
```

## License
Please note that all codes are protected by patents. They can only be used for research purposes. 

## Acknowledgements
Part of the code is adopted from some previous work: [GMStereo](https://github.com/autonomousvision/unimatch). We thank the original authors for their awesome repos. 

## Citation
If you find this project helpful in your research, welcome to cite the paper. 
```
@article{Wang2024DFGANet,
  title = {DFGA-Net: efficient stereo matching with high accuracy via dual-features and gated attention},
  author = {Guohui Wang, Yuanwei Bi, Lujian Zhang, Dawei Wang},
  year = {2024},
  note = {to be published.} 
  }
```
