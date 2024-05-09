from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import h5py
def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.h5_files = self._get_h5_file_list(split)
        self.data, self.labels = self._load_h5_files(self.h5_files)
        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '/root/autodl-tmp/combined/pointnet.pytorch-master-3.16/misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def _get_h5_file_list(self, split):
        file_list_path = os.path.join(self.root, '{}_files.txt'.format(split))
        assert os.path.exists(file_list_path), "Split file list not found: {}".format(file_list_path)
        with open(file_list_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def _load_h5_files(self, h5_files):
        data = []
        labels = []
        for file_path in h5_files:
            with h5py.File(file_path, 'r') as h5_file:
                data.append(h5_file['data'][:])
                labels.append(h5_file['label'][:].squeeze())  # 假设标签在'label'键下并移除单维度
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        return data, labels

    def __getitem__(self, index):
        point_set = self.data[index]
        label = self.labels[index]

        # 数据增强：随机旋转和抖动
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        point_set = torch.from_numpy(point_set.astype(np.float32))
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        return point_set, label

    def __len__(self):
        return self.data.shape[0]

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class ScanObjectNNDataset(Dataset):
    def __init__(self, root, npoints=2048, split='train', data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.h5_files = self._get_h5_file_list(split)
        self.data, self.labels = self._load_h5_files(self.h5_files)
        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '/root/autodl-tmp/combined/pointnet.pytorch-master-3.16/misc/scanobjectnn_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def _get_h5_file_list(self, split):
        file_list_path = os.path.join(self.root, '{}_files.txt'.format(split))
        assert os.path.exists(file_list_path), "Split file list not found: {}".format(file_list_path)
        with open(file_list_path, 'r') as file:
            return [line.strip() for line in file.readlines()]

    def _load_h5_files(self, h5_files):
        data = []
        labels = []
        for file_path in h5_files:
            with h5py.File(file_path, 'r') as h5_file:
                data.append(h5_file['data'][:]) # 假设点云数据在'data'键下
                labels.append(h5_file['label'][:].squeeze()) # 假设标签在'label'键下并移除单维度
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        return data, labels

    def __getitem__(self, index):
        point_set = self.data[index]
        label = self.labels[index]

        # 数据增强：随机旋转、抖动和可能的缩放
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            point_set[:, :3] = point_set[:, :3].dot(rotation_matrix) # 只旋转XYZ，不旋转其他特征
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # 抖动
            # 可以添加更多的数据增强策略，如缩放等

        point_set = torch.from_numpy(point_set.astype(np.float32))
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        return point_set, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

    if dataset == 'scanobjectnn':
        gen_modelnet_id(datapath)
        d = ScanObjectNNDataset(root=datapath)
        print(len(d))
        print(d[0])

