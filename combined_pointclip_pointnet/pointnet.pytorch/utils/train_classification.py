from __future__ import print_function
import argparse
import os
import random
from pointclip_sch.datasets.modelnet40 import ModelNet40
from pointclip_sch.datasets.scanobjnn import ScanObjectNN
from pointclip_sch.trainers.zeroshot import PointCLIP_ZS
import numpy
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dassl.config import get_cfg_default

from dassl.engine import build_trainer

from pointnet.dataset import ShapeNetDataset, ModelNetDataset,ScanObjectNNDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

def reset_cfg(cfg, args):
    if args.dataset:
        cfg.DATASET.ROOT = args.dataset

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    cfg.TRAINER.EXTRA = CN()
def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=100, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=10)
parser.add_argument(
    '--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='/root/autodl-tmp/combined/pointnet.pytorch/cls', help='output folder')
parser.add_argument('--model', type=str, default='/root/autodl-tmp/combined/pointnet.pytorch/cls_model_199_scan_519.pth',help='model path')
parser.add_argument('--dataset', type=str, default='/root/autodl-tmp/combined/data/scanobjectnn',help="dataset path")
parser.add_argument('--dataset_type', type=str, default='scanobjectnn', help="dataset type shapenet|modelnet40|scanobjectnn")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument(
    '--zero-shot', action='store_true', help='zero-shot only', default=True
)
parser.add_argument(
    '--output-dir', type=str, default='output/PointCLIP_ZS/rn50/scanobjectnn', help='output directory'
)
parser.add_argument(
    '--dataset-config-file',
    type=str,
    # default='configs/datasets/scanobjectnn.yaml',
    default='/root/autodl-tmp/combined/pointclip_sch/configs/datasets/scanobjectnn.yaml',
    help='path to config file for dataset setup'
)
parser.add_argument(
    '--config-file', type=str, default='/root/autodl-tmp/combined/pointclip_sch/configs/trainers/PointCLIP_ZS/rn50.yaml', help='path to config file'
)
parser.add_argument(
    '--trainer', type=str, default='PointCLIP_ZS', help='name of trainer'
)
parser.add_argument(
    '--backbone', type=str, default='', help='name of CNN backbone'
)
parser.add_argument(
    '--resume',
    type=str,
    default='',
    help='checkpoint directory (from which the training resumes)'
)
parser.add_argument(
    '--seed',
    type=int,
    default=2,
    help='only positive value enables a fixed seed'
)
parser.add_argument(
    'opts',
    default=None,
    nargs=argparse.REMAINDER,
    help='modify config options using the command-line'
)


opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)


elif opt.dataset_type == 'scanobjectnn':
    dataset = ScanObjectNNDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train')

    test_dataset = ScanObjectNNDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
#调用pointclip的zeroshot模块
cfg = setup_cfg(opt)
trainer = build_trainer(cfg)
pred_pointclip_train = trainer.test_zs(split="train")
pred_pointclip_test = trainer.test_zs(split="test")
if  opt.model == '':


    num_batch = len(dataset) / opt.batchSize
    for epoch in range(opt.nepoch):
        scheduler.step()
        # 训练集累加器
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        classifier.train()  # 设置模型为训练模式
        for i, data in enumerate(dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            pred, trans, trans_feat = classifier(points)
            combined =(pred*0.6 + pred_pointclip_train[i]*0.4)

            loss = F.nll_loss(combined, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * target.size(0)
            pred_choice = combined.data.max(1)[1]

            train_correct += pred_choice.eq(target.data).cpu().sum()
            total_train += target.size(0)

        train_loss_avg = train_loss / total_train
        train_acc_avg = train_correct.float() / total_train

        # 测试集累加器
        test_loss = 0.0
        test_correct = 0
        total_test = 0

        classifier.eval()  # 设置模型为评估模式
        with torch.no_grad():
            for j, data in enumerate(testdataloader, 0):
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                pred, _, _ = classifier(points)
                commbined_test = (pred*0.6+pred_pointclip_test[j]*0.4)
                loss = F.nll_loss(commbined_test, target)

                test_loss += loss.item() * target.size(0)
                pred_choice = commbined_test.data.max(1)[1]
                test_correct += pred_choice.eq(target.data).cpu().sum()
                total_test += target.size(0)

        test_loss_avg = test_loss / total_test
        test_acc_avg = test_correct.float() / total_test

        # 每个epoch结束时打印平均损失和平均准确率
        print('[Epoch %d] Train loss: %.3f, accuracy: %.3f' % (epoch, train_loss_avg, train_acc_avg))
        print('[Epoch %d] Test loss: %.3f, accuracy: %.3f' % (epoch, test_loss_avg, test_acc_avg))

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
#import pandas as pd
# 读取CSV文件
# CSV文件路径
#file_path = '../utils/modified_outputs_modelnet40.csv'

# 确定CSV文件的总列数
# 这里假设所有行具有相同的列数，因此只读取第一行即可
#total_columns = len(pd.read_csv(file_path, nrows=1).columns)
##pointclip_data = pd.read_csv(file_path)

# Assuming testdataloader, test_dataset, pointclip_data, and classifier are defined
#max_accuracy = 0  # Initialize the maximum accuracy variable


total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    combined_val = (pred * 0.6+ pred_pointclip_test[i] * 0.4)
    device = combined_val.device
    pred_choice = combined_val.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("combined(training+testing) final accuracy {}".format(total_correct / float(total_testset)))
total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]





