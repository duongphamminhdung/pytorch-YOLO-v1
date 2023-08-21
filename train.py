import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="deprecated")

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from pytorchsummary import summary
from tqdm import tqdm

from resnet_yolo import resnet50
from yoloLoss import yoloLoss
from dataset import yoloDataset
from config import opt


use_gpu = torch.cuda.is_available()

num_epochs = opt.epoch
batch_size = opt.batch_size
learning_rate = opt.lr_
test_num_workers = opt.test_num_workers
num_workers = opt.num_workers

net = resnet50()
print('model construct complete')

if not os.path.isfile(opt.load_model_path):
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
    print('create model from scratch')
else:
    net.load_state_dict(torch.load(opt.load_model_path))
    print('loaded best model')
summary(model=net, input_size=(3, 640, 640))
# import pdb; pdb.set_trace()
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yoloLoss(7, 2, 5, 0.5)

if use_gpu:
    net.cuda()

net.train()
# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
# optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

train_dataset = yoloDataset(root=opt.root,train=True,transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

test_dataset = yoloDataset(root=opt.root, train=False,transform = [transforms.ToTensor()] )
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=test_num_workers)

print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')
# import pdb; pdb.set_trace()
num_iter = 0
best_test_loss = opt.best_loss
for epoch in range(num_epochs):
    net.train()
    # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.
    print("Training")
    for i,(images,target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = net(images)
        loss = criterion(pred,target)
        total_loss += loss.data
        # total_loss += loss.data[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if (i+1) % 5 == 0:
        #     print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
        #     %(epoch+1, num_epochs, i+1, len(train_loader), loss.data, total_loss / (i+1)))
        #     num_iter += 1
    print("Training loss:", total_loss)
    #validation
    validation_loss = 0.0
    net.eval()
    print("Evaluating")
    for i,(images,target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = Variable(images,volatile=True)
        target = Variable(target,volatile=True)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = net(images)
        loss = criterion(pred,target)
        # validation_loss += loss.data[0]
        validation_loss += loss.data
    # validation_loss /= len(test_loader)
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
    logfile.flush()      
    if not os.path.isdir('outputs'):
        os.mkdir("outupts")
    torch.save(net.state_dict(),'outputs/latest_model.pth')
    print(validation_loss) 
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('Lowest loss %.5f' % best_test_loss)
        # print('Accuracy %.5f' % accuracy)
        torch.save(net.state_dict(),'outputs/best_model_%.5f' % (best_test_loss))
        print("Saved best model")
 