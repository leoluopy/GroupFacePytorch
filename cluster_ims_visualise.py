import pickle
import shutil

import cv2, sys, os

if __name__ == '__main__':
    out_path = "out"

    with open("kmeanGroups.pkl", "rb") as f:
        file_group_dict = pickle.load(f)

    group_file_dict = {}
    for k, v in file_group_dict.items():
        if group_file_dict.get(v) is None:
            group_file_dict[v] = [k]
        else:
            group_file_dict[v].append(k)

    max_group_show_cnt = 50
    for k, v in group_file_dict.items():
        cnt = 0
        for im in v:
            out_full_path = os.path.join(out_path, str(k), os.path.basename(im))
            if os.path.exists(os.path.dirname(out_full_path)) is False:
                os.makedirs(os.path.dirname(out_full_path))
            shutil.copy(im, out_full_path)
            cnt += 1
            if cnt > 5:
                break
