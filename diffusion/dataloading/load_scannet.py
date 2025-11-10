import os
import numpy as np

import torch
from torch.utils.data import Dataset

class ScanNetData(Dataset):
    def __init__(self, path, transform=None, n_points=5120):
        self.path = path
        self.transform = transform
        self.data = []
        scene_list = os.listdir(path)
        scene_list = [scene for scene in scene_list if 'scene' in scene]

        self.center_xy = True
        self.load_data(scene_list)

        self.N_points = n_points

    def load_data(self, scene_list):
        for scene in scene_list:
            data_i = {}

            # load from npz
            npz_path = os.path.join(self.path, scene, 'point_cloud.npz')
            data = np.load(npz_path)
            points = torch.tensor(data['points'], dtype=torch.float32)

            # center xy
            if self.center_xy:
                centroid = torch.mean(points, dim=0)
                points[:, :2] -= centroid[:2]

            data_i['points'] = points

            self.data.append(data_i)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        data = {}
        # select a random point cloud
        random_idx = torch.randint(0, len(self.data), (1,))
        pcd = self.data[random_idx]['points'].clone()

        # subsample point cloud
        random_points_idx = torch.randint(0, len(pcd), (self.N_points,))
        
        data['pcd'] = pcd[random_points_idx]

        rand_scale = True
        rand_trans = True
        # random scale with 0.5 to 1.5
        if rand_scale:
            scale = 0.5 + torch.rand(1)
            data['pcd'] *= scale
        # random translation 
        if rand_trans:
            trans = torch.randn(3)
            data['pcd'] += trans

        if self.transform is not None:
            data = self.transform(data)

        return data