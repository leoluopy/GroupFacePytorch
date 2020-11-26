import sys, os

import cv2
import torch
import numpy as np
import torch.utils.data.dataset
from tqdm import tqdm


def default_loader(bgrImg224):
    input = torch.zeros(1, 3, 224, 224)
    img = bgrImg224
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = torch.from_numpy(img).float()
    input[0, :, :, :] = img
    return input


def torch_loader(bgrImg224):
    if torch.cuda.is_available():
        img = torch.from_numpy(bgrImg224).cuda().float()
    else:
        img = torch.from_numpy(bgrImg224).float()
    img = img.transpose(2, 0).transpose(1, 2) / 255.
    # img = img.unsqueeze(0)  eval 时候需要，train 时候不需要
    return img


class IDDataSet():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = []
        self.file_IDs = []
        self.file_labels = []

        self.IDs = []
        self.IDsLabels = {}

        for dir in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, dir)) is False:
                raise ("DIR Error")
            self.IDs.append(dir)

            label_idx = 0
            for ID in self.IDs:
                self.IDsLabels[ID] = label_idx
                label_idx += 1

        for dir in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, dir)) is False:
                raise ("DIR Error")
            for file in os.listdir(os.path.join(self.root_dir, dir)):
                if os.path.splitext(file)[1] in [".jpg", ".bmp", ".png"]:
                    self.file_paths.append(os.path.join(self.root_dir, dir, file))
                    self.file_IDs.append(dir)
                    self.file_labels.append(self.IDsLabels[dir])

        return

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_id = self.file_IDs[idx]
        file_label = self.file_labels[idx]
        return torch_loader(cv2.imread(file_path)), file_path, file_id, file_label

    def __len__(self):
        return len(self.file_paths)


if __name__ == '__main__':
    # img = default_loader(cv2.imread("../demo_ims/0000166/260.jpg"))
    # img = torch_loader(cv2.imread("../demo_ims/0000166/260.jpg"))

    dataset = IDDataSet("../demo_ims/")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    for img, file_path, id, label in tqdm(data_loader):
        bgrIm = (img[0] * 255.0).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        cv2.imshow("x", bgrIm)
        cv2.waitKey(0)
        pass
    pass
