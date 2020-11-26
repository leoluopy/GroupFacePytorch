import sys, os
import torch
from models.group_face import GroupFace
from loss.arcface_loss import ArcFaceLoss
from system.data_loader import IDDataSet


def train(model, epoch):
    model.train()

    batch_len = len(data_loader)
    loss_sum = 0.0
    for i, (img, file_path, id, label) in enumerate(data_loader):
        img = img.cuda()
        label = label.cuda()
        group_inter, final, group_prob, group_label = model(img)
        loss = criteria_arc(final, label)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_sum += float(loss)
        print("Epoch: {} [{}/{}] loss:{:.2f}".format(epoch, i, batch_len, float(loss)))

    print("Epoch {} Trained Loss_Sum:{:.5f}".format(epoch, loss_sum / float(batch_len)))


if __name__ == '__main__':

    epoch_whole = 200

    dataset = IDDataSet("./demo_ims/")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = GroupFace(resnet=18)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criteria_arc = ArcFaceLoss()
    criteria_arc = torch.nn.CrossEntropyLoss()

    print("     START Training")
    for epoch in range(epoch_whole):
        train(model, epoch)

    pass
