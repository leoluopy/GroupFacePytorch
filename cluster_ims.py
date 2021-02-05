import pickle
import random
import shutil
import threading
import time
from os import cpu_count

import numpy as np
import sys, os, cv2

import threadpool
import torch
import torch.utils.data
from skimage.feature import local_binary_pattern
from kmeans_pytorch import kmeans, kmeans_predict
from tqdm import tqdm
import jpeg4py

root_path = "./cluster_im/"
groups = 3
batch_size = 8


def ThreadPool(PoolCnt):
    pool = threadpool.ThreadPool(PoolCnt)
    return pool


def ThreadSchedule(pool, enter_func, param_list):
    requests = threadpool.makeRequests(enter_func, param_list)
    [pool.putRequest(req) for req in requests]


def GetFeat(path, feat_all, mutex, idx):
    try:
        try:
            image = jpeg4py.JPEG(path).decode()
        except Exception as ex:
            image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(image, 8, 1)
        feat = np.histogram(lbp, 128)[0]
        feat = torch.tensor(feat).unsqueeze(0)
        if torch.cuda.is_available():
            feat = feat.cuda()
        CatFeat(mutex, feat, feat_all, idx)
    except Exception as ex:
        print(str(ex))
        print("loading Error: {}".format(path))


def CatFeat(mutex, feat, feat_all, idx):
    # 锁定
    mutex.acquire()
    feat_all[idx] = feat
    # 释放
    mutex.release()


def CheckFinished(mutex, feat_all, len):
    ret = False
    # 锁定
    mutex.acquire()
    cur_len = feat_all.__len__()
    if cur_len < len:
        ret = False
    else:
        ret = True
    # 释放
    mutex.release()
    return ret


def BatchLBPGenerater(pathes, batch_size, pool):
    start_idx, end_idx = 0, 0
    mutex = threading.Lock()
    feat_all = {}
    feat_batch = {}
    ts = []
    cur_batch_len = 0
    while start_idx < len(pathes) - 1:
        ts.clear()
        feat_all.clear()
        end_idx = batch_size + start_idx if (batch_size + start_idx) < len(pathes) else len(pathes)
        for i, path in enumerate(pathes[start_idx:end_idx]):
            # t = threading.Thread(target=GetFeat, args=(path, feat_all, mutex, i,))
            ThreadSchedule(pool, GetFeat, [((path, feat_all, mutex, i), None)])

        cur_batch_len = len(pathes[start_idx:end_idx])
        while CheckFinished(mutex, feat_all, cur_batch_len) is False:
            time.sleep(0.01)
        for i in range(cur_batch_len):
            if i == 0:
                feat_batch = feat_all[i]
            else:
                feat_batch = torch.cat((feat_batch, feat_all[i]), 0)
            feat_all[i].detach()

        start_idx = end_idx
        yield feat_batch


def GetAllFilesFeat(file_paths):
    file_feats = {}
    pool = ThreadPool(batch_size)
    lbp_loader = BatchLBPGenerater(file_paths, batch_size, pool)

    cnt = 0
    wholeLen = int(len(file_paths) / batch_size)
    for feat in lbp_loader:
        sys.stdout.write("\r >> {}/{}".format(cnt, wholeLen))
        sys.stdout.flush()
        cnt += 1
        if isinstance(file_feats, dict) is True:
            file_feats = feat
        else:
            file_feats = torch.cat((file_feats, feat), 0)
        feat.detach()
    return file_feats


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    dims, num_clusters = 128, groups

    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1] in ['.jpg', '.png']:
                full_path = os.path.join(root, file)
                if os.path.getsize(full_path) > 0:
                    file_paths.append(full_path)
                else:
                    print("drop file :{}".format(full_path))

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
