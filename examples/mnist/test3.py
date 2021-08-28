from torchvision import datasets, transforms, models
from torch import nn, optim, utils, device as device_, cuda
import torch
import numpy as np
from sklearn import metrics
import time
import sparseconvnet as scn

dataset_train = datasets.MNIST(
    './data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor())
dataset_valid = datasets.MNIST(
    './data', 
    train=False, 
    download=True, 
    transform=transforms.ToTensor())

dataloader_train = utils.data.DataLoader(dataset_train,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)
dataloader_valid = utils.data.DataLoader(dataset_valid,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)


lr = 0.01

device = device_("cuda" if cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        # super(Model, self).__init__()
        # self.conv1 = nn.Conv2d(1, 64, 5) # -> 24x24
        # self.pool1 = nn.MaxPool2d(2) # -> 12x12
        # self.conv2 = nn.Conv2d(64, 128, 5) # -> 8x8
        # self.dropout = nn.Dropout(p=0.4)
        # self.dense = nn.Linear(128 * 8 * 8, 10)
        
        # nn.Module.__init__(self)
        # self.sparseModel = scn.Sequential(
        #     #scn.SubmanifoldConvolution(2, 1, 8, 3, False),
        #     scn.Convolution(2, 1, 64, 5, 1, False),
        #     scn.MaxPooling(2, 2, 2), # MaxPool2d, (2), 2 stride
        #     scn.Convolution(2, 64, 128, 5, 1, False),
        #     scn.SparseToDense(2, 64))
        # self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
        # self.inputLayer = scn.InputLayer(2,self.spatial_size,2)
        # self.linear = nn.Linear(64, 10)

        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(3, torch.LongTensor([28*2+2]*3), mode=2)).add(
           scn.Convolution(2, 1, 64, 5, 1, False)).add(
           scn.MaxPooling(2, 2, 2)).add(
           scn.Convolution(2, 64, 128, 5, 1, False)).add(
           scn.SparseToDense(2, 64))
        self.linear = nn.Linear(64, 10)

# Model(
#   (conv1): Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1))
#   (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (conv2): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
#   (dropout): Dropout(p=0.4, inplace=False)
#   (dense): Linear(in_features=8192, out_features=10, bias=True)
# )


    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        # x = F.relu(self.conv2(x))
        # x = self.dropout(x)
        # x = x.view(x.size(0), -1) # Flatten
        # return F.relu(self.dense(x))
        x = self.sparseModel(x)
        x = self.linear(x)
        return x
    
model = Model().to(device)
scale=63
# dataset = {'train' : dataloader_train, 'val' : dataloader_valid}


print(model)

# scn.ClassificationTrainValidate(
#     model, dataset,
#     {'n_epochs': 10,
#      'initial_lr': 0.1,
#      'lr_decay': 0.05,
#      'weight_decay': 1e-4,
#      'use_cuda': torch.cuda.is_available(),
#      'check_point': False, })


optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.perf_counter()

model.train()
for i in range(5):
  print(i)
  for x, t in dataloader_train:
      x = x.to(device)
      t = t.to(device)
      model.zero_grad()
      y = model(x)
      loss = criterion(y, t)
      loss.backward()
      optimizer.step()

end = time.perf_counter()
print(f"time = {end - start: 0.4f} seconds")



# model.eval()
# labels = []
# preds = []
# losses = []
# for x, t in dataloader_valid:
#     x = x.to(device)
#     t = t.to(device)
#     labels.extend(t.tolist())
#     y = model(x)
#     loss = criterion(y, t)
#     losses.append(loss.cpu().data)
#     pred = y.argmax(1)
#     preds.extend(pred.tolist())
# print('Loss: {:.3f}, Accuracy: {:.3f}'.format(
#     np.mean(losses),
#     metrics.accuracy_score(labels, preds, normalize=True)
# ))