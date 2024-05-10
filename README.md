# Hechengjun's graduation poject


## Introduction
本项目包含本科毕业项目涉及到的所有代码和数据集，主要内容有：PointNet、PointNet++、PointCLIP模型的分类算法复现以及PointCLIP模型zeroshot模块与PointNet模型结合在ScanObjectNN数据集上的分类表现上的优化。




## Requirements

### Installation
PointNet和PointNet++所需环境安装：
PyTorch 1.6.0、Python 3.8、CUDA 10.1  requirements去系统里看一下
```bash
conda create -n pointnet python=3.8
conda activate pointnet
pip install -r requirements.txt
```


Create a conda environment and install dependencies:
```bash
cd PointCLIP
conda create -n pointclip python=3.8
conda activate pointclip
pip install -r requirements.txt
```
# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
PyTorch 1.9.0 和 CUDA 11.1

# Install the modified dassl library (no need to re-build if the source code is changed)
cd Dassl3D/
python setup.py develop

cd ..
```

### Dataset
Download the official [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) dataset and put the unzip folder under `data/`.
The directory structure should be:
```bash
│PointCLIP/
├──...
├──data/
│   ├──modelnet40_ply_hdf5_2048/
├──...
```
### 

## Get Started

### Zero-shot PointCLIP
Edit the running settings in `scripts/zeroshot.sh`, e.g. config file and output directory. Then run Zero-shot PointCLIP:
```bash
cd scripts
bash zeroshot.sh
```
If you need the post-search for the best view weights, add `--post-search` and modulate the search parameters in the config file. More search time leads to higher search results but longer time.

### Few-shot PointCLIP
Set the shot number and other settings in `scripts/fewshot.sh`. Then run PointCLIP with the inter-view adapter:
```bash
cd scripts
bash fewshot.sh
```
`--post-search` is also optional.

### Evaluation
Download the pre-pretrained [checkpoint](https://drive.google.com/file/d/1hFswVidomLdYaWZZga6RpWRJW9-JJbHZ/view?usp=sharing) by 16-shot fine-tuning and put it under `ckpt/adapter/`. It will produce 86.71% on ModelNet40 test set and 87%+ by post-search:
```bash
cd scripts
bash eval.sh
```
You can edit the `--model-dir` and `--output-dir` to evaluate checkpoints trained by your own.

## Acknowlegment
This repo benefits from [CLIP](https://github.com/openai/CLIP), [SimpleView](https://github.com/princeton-vl/SimpleView) and the excellent codebase [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). Thanks for their wonderful works.

## Citation
```bash
@article{zhang2021pointclip,
  title={PointCLIP: Point Cloud Understanding by CLIP},
  author={Zhang, Renrui and Guo, Ziyu and Zhang, Wei and Li, Kunchang and Miao, Xupeng and Cui, Bin and Qiao, Yu and Gao, Peng and Li, Hongsheng},
  journal={arXiv preprint arXiv:2112.02413},
  year={2021}
}
```

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
