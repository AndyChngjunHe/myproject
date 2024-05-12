# Hechengjun's graduation poject


## Introduction
本项目包含本科毕业项目涉及到的所有代码和数据集，主要内容有：PointNet、PointNet++、PointCLIP模型的分类算法复现以及PointCLIP模型zeroshot模块与PointNet模型结合在ScanObjectNN数据集上的分类表现上的优化。




## Requirements

### Installation
PointNet和PointNet++所需环境安装：
```bash
conda create -n pointnet python=3.8
conda activate pointnet
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```


PointCLIP所需环境安装:
```bash
conda create -n pointclip python=3.8
conda activate pointclip
conda install pytorch==1.9.0 cudatoolkit=11.1 -c pytorch
pip install -r requirements.txt

# Install the modified dassl library (no need to re-build if the source code is changed)
cd Dassl3D/
python setup.py develop

```

### Dataset
Download the official [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) dataset and put the unzip folder under `data/`.
Download the official [ScanObjectNN](https://docs.google.com/forms/d/e/1FAIpQLSeHkKPspO4NyVozXkCMOv4UuvXpn2Qb3WG3_3AILFcRni9ArQ/viewform) dataset and put the unzip folder under `data/`.
scanobjectnn数据集请使用项目代码中data/scanobjectnn/中的train_files.txt和test_files.txt文件
数据集在各模型中位置如下：
│pointnet.pytorch-master-3.16/
├──...
├──data/
│   ├──modelnet40_ply_hdf5_2048/
│   ├──scanobjectnn/
├──...

│Pointnet_Pointnet2_pytorch-master/
├──...
├──data/
│   ├──modelnet40_ply_hdf5_2048/
│   ├──scanobjectnn/
├──...


│PointCLIP/
├──...
├──data/
│   ├──modelnet40_ply_hdf5_2048/
│   ├──scanobjectnn/
├──...


## Get Started
### PointNet Classification
Training
```bash
cd utils
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | scanobjectnn>

```
Use --feature_transform to use feature transform.

### PointNet++
If you want to train on ModelNet40, you can use --num_category 40.
If you want to train on ScanObjectNN, you can use --num_category 15.
```bash
# ModelNet40
## Select different models in ./models 

python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --num_category 40
python test_classification.py --log_dir pointnet2_cls_ssg --num_category 40
python train_classification.py --model pointnet2_cls_msg --log_dir pointnet2_cls_msg --num_category 40
python test_classification.py --log_dir pointnet2_cls_msg --num_category 40

# ScanObjectNN
## Select different models in ./models 
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --num_category 15
python test_classification.py --log_dir pointnet2_cls_ssg --num_category 15
python train_classification.py --model pointnet2_cls_msg --log_dir pointnet2_cls_msg --num_category 15
python test_classification.py --log_dir pointnet2_cls_msg --num_category 15

```

### Zero-shot PointCLIP
#### 需要修改数据集或者使用的模型，可以在zeroshot.sh和fewshot.sh中修改参数
Edit the running settings in `scripts/zeroshot.sh`, e.g. config file and output directory. Then run Zero-shot PointCLIP:
```bash
cd scripts
bash zeroshot.sh
```


### Few-shot PointCLIP
Set the shot number and other settings in `scripts/fewshot.sh`. Then run PointCLIP with the inter-view adapter:
```bash
cd scripts
bash fewshot.sh
```


## Acknowlegment
This repo benefits from [CLIP](https://github.com/openai/CLIP), [SimpleView](https://github.com/princeton-vl/SimpleView) and the excellent codebase [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). Thanks for their wonderful works.


