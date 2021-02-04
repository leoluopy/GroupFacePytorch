import pickle
import random
import shutil

import numpy as np
import sys, os, cv2
import torch
from skimage.feature import local_binary_pattern
from kmeans_pytorch import kmeans, kmeans_predict
from tqdm import tqdm
import jpeg4py

root_path = "./cluster_im/"



def GetAllFilesFeat(file_paths):
    file_feats = {}
    for file in tqdm(file_paths):
        try:
            image = jpeg4py.JPEG(file).decode()
        except Exception as ex:
            image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(image, 8, 1)
        feat = np.histogram(lbp, 128)[0]
        feat = torch.tensor(feat).unsqueeze(0).cuda()
        if isinstance(file_feats, dict) is True:
            file_feats = feat
        else:
            file_feats = torch.cat((file_feats, feat), 0)
    return file_feats


if __name__ == '__main__':

    dims, num_clusters = 128, 32

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
