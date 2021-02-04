from os import cpu_count

import cv2
import os
import pickle

import jpeg4py
import numpy as np
import torch
import torch.utils.data
from kmeans_pytorch import kmeans
from skimage.feature import local_binary_pattern
from tqdm import tqdm

root_path = "./cluster_im/"


class BatchLBPLoader(torch.utils.data.Dataset):
    def __init__(self, pathes):
        super(BatchLBPLoader, self).__init__()
        self.pathes = pathes

    def __getitem__(self, idx):
        file = self.pathes[idx]
        try:
            image = jpeg4py.JPEG(file).decode()
        except Exception as ex:
            image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(image, 8, 1)
        feat = np.histogram(lbp, 128)[0]
        if torch.cuda.is_available():
            feat = torch.tensor(feat).cuda()
        else:
            feat = torch.tensor(feat)
        return feat

    def __len__(self):
        return len(self.pathes)


def GetAllFilesFeat(file_paths):
    file_feats = {}
    lbp_set = BatchLBPLoader(file_paths)
    lbp_loader = torch.utils.data.DataLoader(lbp_set, batch_size=8, shuffle=False, num_workers=int(cpu_count() / 2))
    # lbp_loader = torch.utils.data.DataLoader(lbp_set, batch_size=8, shuffle=False, num_workers=0)

    for feat in tqdm(lbp_loader):
        if isinstance(file_feats, dict) is True:
            file_feats = feat
        else:
            file_feats = torch.cat((file_feats, feat), 0)
    return file_feats


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    dims, num_clusters = 128, 3

    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1] in ['.jpg', '.png']:
                file_paths.append(os.path.join(root, file))

    print("file path len: {}".format(len(file_paths)))
    file_feats = GetAllFilesFeat(file_paths)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # k-means
    cluster_ids_x, cluster_centers = kmeans(
        X=file_feats, num_clusters=num_clusters, distance='euclidean', device=device
    )
    assert (cluster_ids_x.shape[0] == file_feats.shape[0])

    file_group_dict = {}

    for i, file in enumerate(file_paths):
        file_group_dict[file] = int(cluster_ids_x[i])

    with open("kmeanGroups.pkl", "wb") as f:
        pickle.dump(file_group_dict, f)
    print("DONE")
