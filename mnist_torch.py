import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

def load_dataset(path):
    all_data = pd.read_csv(path + "/train.csv")
    all_image = all_data.values[:, 1:].reshape(all_data.shape[0], 28, 28)
    all_label = all_data.values[:, 0]
    train_image = all_image[:int(all_image.shape[0] * 0.8)]
    train_label = all_label[:int(all_image.shape[0] * 0.8)]
    valid_image = all_image[int(all_image.shape[0] * 0.8):]
    valid_label = all_label[int(all_image.shape[0] * 0.8):]
    return train_image, train_label, valid_image, valid_label

def make_label(label):
    one_hot = np.zeros((label.shape[0], 10))
    for i in range(label.shape[0]):
        one_hot[i][label[i]] = 1
    return one_hot
    
VISUALIZE = False
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        # self.fc1 = Linear(32, 10)
        self.fc1 = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        if VISUALIZE:
            x1 = x.detach()
            fig, axes = plt.subplots(1, 1)
            # axes = axes.flatten()
            axes.imshow(x1[0][0])
            fig.suptitle(f"Min: {x1[0].min(): .2f}, Max: {x1[0].max(): .2f}")
            plt.show()
        x = self.conv1(x).relu()
        x = F.max_pool2d(x, 2)
        if VISUALIZE:
            x1 = x.detach()
            fig, axes = plt.subplots(2, 2)
            axes = axes.flatten()
            for i in range(4):
                axes[i].imshow(x1[0][i])
                axes[i].set_title(f"Min: {x1[0][i].min(): .2f}, Max: {x1[0][i].max(): .2f}")
            fig.suptitle(f"Min: {x1[0].min(): .2f}, Max: {x1[0].max(): .2f}")
            plt.show()
        x = self.conv2(x).relu()
        x = F.max_pool2d(x, 2)
        if VISUALIZE:
            x1 = x.detach()
            fig, axes = plt.subplots(4, 4)
            axes = axes.flatten()
            for i in range(16):
                axes[i].imshow(x1[0][i])
                axes[i].set_title(f"Min: {x1[0][i].min(): .2f}, Max: {x1[0][i].max(): .2f}")
            fig.suptitle(f"Min: {x1[0].min(): .2f}, Max: {x1[0].max(): .2f}")
            plt.show()
        x = self.conv3(x).relu()
        x = x.reshape((*x.shape[:-3], x.shape[-1] * x.shape[-2] * x.shape[-3]))
        # x = x.mean((-1, -2))
        x = self.fc1(x)
        return x, x.sigmoid()

class Dataset:
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        images, labels = self.images[idx * self.batch_size:(idx + 1) * self.batch_size], self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels = make_label(labels)
        images, labels = Tensor(images).reshape((images.shape[0], 1, *images.shape[1:])) / 255, Tensor(labels)
        return images, labels

EPOCH = 100
model = CNN()
optimizer = optim.SGD(model.parameters(), lr=0.1)
ti, tl, vi, vl = load_dataset("~/Datasets/MNIST")
# ti, tl = ti[:100], tl[:100]
train_set = Dataset(ti, tl, 16)
valid_set = Dataset(vi, vl, 16)

def visualize(images, labels):
    fig, axes = plt.subplots(4, 4)
    axes = axes.flatten()
    for i in range(images.shape[0]):
        axes[i].imshow(images[i][0], cmap="gray")
        axes[i].set_title(labels[i].argmax(-1))
    plt.show()

for epoch in range(1):
    for i in range(4000):
    # for i in range(len(train_set)):
        VISUALIZE = (i % 200 == 0)
        images, labels = train_set[i]
        if VISUALIZE: visualize(images, labels)
        logits, pred = model(images)
        pred.retain_grad()
        # loss = cross_entropy_loss(logits, labels)
        loss = F.mse_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        # print(model.conv1.weights.sum().item())
        print(labels[0])
        print(pred[0])
        print(pred.grad[0])
        # print(logits.softmax(-1).buffer.data[0])
    # print("Validating...")
    # print("----")
    # for i in range(len(valid_set)):
    #     images, labels = valid_set[i]
    #     pred = model(images)
    #     loss = mse_loss(pred, labels)
    #     print(loss.item())
