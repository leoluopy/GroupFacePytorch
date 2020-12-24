import math

import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # partial fc 时，类中心个数会发生变化, label数值也要对应发生变化。
    def forward(self, inputs, targets, use_label_smoothing=True):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.to(torch.device('cuda'))
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class ArcFaceLoss(nn.Module):
    def __init__(self, m=0.1, s=1.0, d=256, num_classes=10, use_gpu=True, partial_fc_rate=2):
        super(ArcFaceLoss, self).__init__()
        self.m = m
        self.s = s
        self.num_classes = num_classes
        self.partial_fc_rate = partial_fc_rate

        self.weight = torch.nn.Linear(d, num_classes, bias=False)
        if use_gpu:
            self.weight = self.weight.cuda()
        bound = 1 / math.sqrt(d)
        nn.init.uniform_(self.weight.weight, -bound, bound)
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes, use_gpu=use_gpu)

    def get_center_subscript(self, center_Idxs, plabel):
        idx = 0
        for centerId in center_Idxs:
            if centerId == plabel:
                if torch.cuda.is_available():
                    return torch.tensor([idx]).cuda()
                else:
                    return torch.tensor([idx])
            idx += 1
        return -1

    def partial_sample(self, positive_labels):
        centers_Idxs = {}
        new_labels = {}
        p_num = positive_labels.shape[0]
        for i in range(p_num):
            if isinstance(centers_Idxs, dict):
                centers_Idxs = positive_labels[i].reshape(-1)
            elif positive_labels[i] not in centers_Idxs:
                centers_Idxs = torch.cat((centers_Idxs, positive_labels[i].reshape(-1)))
            if isinstance(new_labels, dict):
                new_labels = self.get_center_subscript(centers_Idxs, positive_labels[i])
            else:
                new_labels = torch.cat((new_labels, self.get_center_subscript(centers_Idxs, positive_labels[i])))

        choosed_centers = self.weight.weight[centers_Idxs]
        return choosed_centers, new_labels

    def forward(self, x, labels):
        '''
        x : feature vector : (b x  d) b= batch size d = dimension
        labels : (b,)
        '''
        raw_label = labels
        choosed_centers, new_labels = self.partial_sample(labels)
        labels = new_labels
        with torch.no_grad():
            # self.weight.weight.div_(torch.norm(self.weight.weight, dim=1, keepdim=True))
            choosed_centers.div_(torch.norm(choosed_centers, dim=1, keepdim=True))

        x = nn.functional.normalize(x, p=2, dim=1)  # normalize the features

        b = x.size(0)
        n = self.num_classes

        # cos_angle = self.weight(x)
        cos_angle = torch.matmul(x, choosed_centers.t())
        cos_angle = torch.clamp(cos_angle, min=-1, max=1)
        for i in range(b):
            # cos_angle[i][labels[i]] = torch.cos(torch.acos(cos_angle[i][labels[i]]) + self.m)
            with torch.no_grad():
                delta_cos = torch.cos(torch.acos(cos_angle[i][labels[i]]) + self.m) - cos_angle[i][labels[i]]
            cos_angle[i][labels[i]] = cos_angle[i][labels[i]] + delta_cos
            pass
        weighted_cos_angle = self.s * cos_angle
        log_probs = self.CrossEntropy(weighted_cos_angle, labels)
        return log_probs


def case2():
    criteria = ArcFaceLoss()

    label = torch.tensor(
        [0]).cuda()
    x = torch.rand((1, 3, 224, 224)).cuda()

    from models.group_face import GroupFace
    model = GroupFace(resnet=18)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_center = torch.optim.Adam(criteria.weight.parameters(), lr=1e-4)

    group_inter, final, group_prob, group_label = model(x)
    loss = criteria(final, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer_center.step()

    print("END")


def case1():
    criteria = ArcFaceLoss(num_classes=10000)
    # x = torch.rand(32, 2048).cuda()
    # label = torch.tensor(
    #     [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, ]).cuda()

    x = torch.rand(1, 256, requires_grad=True).cuda()
    label = torch.tensor(
        [0]).cuda()

    loss = criteria(x, label)
    print(loss)


if __name__ == '__main__':
    case1()
    # case2()

    pass
