import os
import numpy as np
import warnings
import h5py
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals

        if split == 'train':
            filepath = os.path.join(self.root, 'train_files.txt')
        else:
            filepath = os.path.join(self.root, 'test_files.txt')

        with open(filepath, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]

        self.datapath = []
        for file in self.file_list:
            self._load_h5_data(os.path.join(self.root, file))

    def _load_h5_data(self, h5_filename):
        with h5py.File(h5_filename, 'r') as h5_file:
            points = h5_file['data'][:]
            labels = h5_file['label'][:]
            for i in range(points.shape[0]):
                self.datapath.append((points[i], labels[i]))

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        point_set, label = self.datapath[index]

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    class Args:
        num_point = 2048
        use_uniform_sample = False
        use_normals = False

    args = Args()
    data = ModelNetDataLoader('/data/scanobjectnn', args, split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
