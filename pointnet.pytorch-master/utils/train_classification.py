from __future__ import print_function
import argparse
import os
import random

import numpy
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset,ScanObjectNNDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=100, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='',help='model path')
parser.add_argument('--dataset', type=str, default='../data/modelnet40_ply_hdf5_2048',help="dataset path")
parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40|scanobjectnn")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

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
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * target.size(0)
            pred_choice = pred.data.max(1)[1]

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
                loss = F.nll_loss(pred, target)

                test_loss += loss.item() * target.size(0)
                pred_choice = pred.data.max(1)[1]
                test_correct += pred_choice.eq(target.data).cpu().sum()
                total_test += target.size(0)

        test_loss_avg = test_loss / total_test
        test_acc_avg = test_correct.float() / total_test

        # 每个epoch结束时打印平均损失和平均准确率
        print('[Epoch %d] Train loss: %.3f, accuracy: %.3f' % (epoch, train_loss_avg, train_acc_avg))
        print('[Epoch %d] Test loss: %.3f, accuracy: %.3f' % (epoch, test_loss_avg, test_acc_avg))

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))


total_correct = 0
total_testset = 0

for i,data in tqdm(enumerate(testdataloader, 0)):

    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    device = pred.device
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))