import os
import torch
import csv
import pickle
import random
import math
import numpy as np
import torch.utils.data as data
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from PIL import Image

class WSI_Dataset(Dataset):
    def __init__(self, list_path, data_path, status='feat'):
        self.status = status
        self.feat_list = []
        self.dist_list = []
        self.label_list = []
        with open(list_path) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                self.feat_list.append(os.path.join(data_path, row[0] + '_feat.pkl'))
                self.dist_list.append(os.path.join(data_path, row[0] + '_dist.pkl'))
                self.label_list.append(row[1])

    def __len__(self):
        return len(self.label_list)
    
    def get_weights(self):
        labels = np.asarray(self.label_list, np.int64)
        tmp = np.bincount(labels)
        weights = 1.0 / np.asarray(tmp[labels], np.float64)

        return weights

    def __getitem__(self, idx):
        with open(self.feat_list[idx], 'rb') as f:
                wsi_data = pickle.load(f)

        features = wsi_data['feats']
        label = int(wsi_data['label'])
        features = torch.from_numpy(features)
        features = torch.flatten(features, 1)

        if self.status == 'feat':
            return features, label, torch.zeros(features.shape), torch.zeros(features.shape)
        elif self.status == 'dist':
            with open(self.dist_list[idx], 'rb') as f:
                dist_data = pickle.load(f)

            mu = dist_data['mu']
            logvar = dist_data['logvar']
            
            return features, label, mu, logvar


class CategoryDataset(data.Dataset):
    def __init__(self, data_path, transforms):
        data_file = open(data_path)

        self.transforms = transforms

        self.images = []
        self.labels = []
    
        try:
            text_lines = data_file.readlines()
            for i in text_lines:
                i = i.strip()
                self.images.append(i.split(' ')[0])
                self.labels.append(int(i.split(' ')[1]))
        finally:
            data_file.close()

    def __getitem__(self, ind):
        image = Image.open(self.images[ind])
        label = self.labels[ind]
        image, do_it_list_1, do_it_list_2 = self.transforms(image)

        return image, label, [do_it_list_1, do_it_list_2]

    def __len__(self):
        return len(self.images)


