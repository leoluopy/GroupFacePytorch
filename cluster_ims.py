import random
import shutil

import numpy as np
import sys, os, cv2
import torch
from skimage.feature import local_binary_pattern
from kmeans_pytorch import kmeans, kmeans_predict
from tqdm import tqdm

if __name__ == '__main__':
    # root_path = "../../ims/demo_ims"
    root_path = "./cluster_im/"
    data_size, dims, num_clusters = 1000, 128, 3

    cluster_out_path = "out"
    if os.path.exists(cluster_out_path) is False:
        os.makedirs(cluster_out_path)

    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1] in ['.jpg', '.png']:
                file_paths.append(os.path.join(root_path, root, file))

    print("file path len: {}".format(len(file_paths)))
    random.shuffle(file_paths)
    file_feats = {}
    for file in tqdm(file_paths):
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        lbp = local_binary_pattern(image, 8, 1)
        feat = np.histogram(lbp, 128)[0]
        feat = torch.tensor(feat).unsqueeze(0).cuda()
        if isinstance(file_feats, dict) is True:
            file_feats = feat
        else:
            file_feats = torch.cat((file_feats, feat), 0)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # k-means
    cluster_ids_x, cluster_centers = kmeans(
        X=file_feats, num_clusters=num_clusters, distance='euclidean', device=device
    )
    assert (cluster_ids_x.shape[0] == file_feats.shape[0])
    cnt = 0
    for id in tqdm(cluster_ids_x):
        out_file_path = os.path.join(cluster_out_path, str(id.item()))
        if os.path.exists(out_file_path) is False:
            os.makedirs(out_file_path)

        shutil.copy(file_paths[cnt], out_file_path)
        cnt += 1
