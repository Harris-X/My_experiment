# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from yolox_model import *
from torch import nn
from torch.utils.data import DataLoader
import torch

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)



def tensor_to_PIL(tensor):
    loader = transforms.Compose([transforms.ToTensor()])
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image




# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


print(len(train_dataloader))
print(len(test_dataloader))

# 创建网络模型
tudui = CSPDarknet()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)
# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 1
# 添加tensorboardC
writer = SummaryWriter("../yolox_logs_train")


for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = data
        #将(64,3,32,32) 转化为（64，3 640，640）

        # for i in range(len(imgs[0])):
        #     img = tensor_to_PIL(imgs[i])
        #     img = img.convert('RGB')
        #     transform = torchvision.transforms.Compose([torchvision.transforms.Resize((640, 640)),
        #                                                 torchvision.transforms.ToTensor()])
        #     img = transform(img)
        #     temp = torch.zeros(64, 3, 640, 640)
        #     temp[i]=img


        #outputs = tudui(temp)    #(3,1024,20,20)
        outputs = tudui(imgs)   #(3,1024,1,1)

        #将损失函数输入从三维降低成一维
        outputs=outputs.view(outputs.size(0),outputs.size(1)*outputs.size(2)*outputs.size(3))
        #outputs = outputs.view(outputs.size(0), outputs.size(1))
        #outputs = outputs.squeeze(2).squeeze(2)

        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data


            # for i in range(len(imgs[0])):
            #     img = tensor_to_PIL(imgs[i])
            #     img = img.convert('RGB')
            #     transform = torchvision.transforms.Compose([torchvision.transforms.Resize((640, 640)),
            #                                                 torchvision.transforms.ToTensor()])
            #     img = transform(img)
            #     temp = torch.zeros(64, 3, 640, 640)
            #     temp[i] = img

            #outputs = tudui(temp)
            outputs = tudui(imgs)

            outputs = outputs.view(outputs.size(0), outputs.size(1) * outputs.size(2) * outputs.size(3))
            #outputs = outputs.view(outputs.size(0), outputs.size(1))
            #outputs = outputs.squeeze(2).squeeze(2)

            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

torch.save(tudui, "tudui_lvk.pth")
print("模型已保存")
writer.close()
