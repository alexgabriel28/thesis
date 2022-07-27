import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader

import tensorflow
import torch
from torch.utils.data import DataLoader, Dataset

from thesis.helper import dataset, utils, tensor_img_transforms
from thesis.models import eval
from thesis.helper.utils import save_train_specs
from sklearn.model_selection import train_test_split

import os
import numpy as np
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image

from sklearn.model_selection import train_test_split

import augly.image as imaugs

from tqdm.auto import tqdm

from thesis.helper import tensor_img_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self,inputs):
        return self.feature(inputs)

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class iCaRLmodel:
    def __init__(
        self,
        numclass,
        feature_extractor,
        batch_size,
        task_size,
        memory_size,
        epochs,
        learning_rate, 
        train_dataloaders, 
        test_dataloaders,
        img_size,
        root_dir,
        run_name,
        *args,
        **kwargs,
        ):

        super(iCaRLmodel, self).__init__()
        self.path = root_dir
        self.run_name = run_name
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.model = network(numclass,feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize(img_size),
                                             #transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None

        self.train_transform = transforms.Compose([transforms.Resize(img_size),
                                                  transforms.RandomCrop((32,32),padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  #transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.test_transform = transforms.Compose([transforms.Resize(img_size),
                                                   #transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.classify_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                    transforms.Resize(img_size),
                                                    #transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.train_dataloaders = train_dataloaders
        self.test_dataloaders = test_dataloaders
        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.acc_lists = []
        self.acc_lists_file = self.path + "_" + self.run_name + "_test_acc.txt"
        with open(self.acc_lists_file, 'w') as fp:
            pass

        self.train_loader=None
        self.test_loader=None

    # get incremental train data
    # incremental
    def beforeTrain(self, train_loader, test_loader):
        self.model.eval()
        classes=[self.numclass-self.task_size,self.numclass]
        self.train_loader, self.test_loader = train_loader, test_loader
        # self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.numclass>self.task_size:
            self.model.Incremental_learning(self.numclass)
        print(self.numclass)
        self.model.train()
        self.model.to(device)

    # def _get_train_and_test_dataloader(self, classes):
    #     train_loader = next(iter(self.train_dataloaders))
    #     test_loader = next(iter(self.test_dataloaders))

    #     return train_loader, test_loader
    
    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    # train model
    # compute loss
    # evaluate model
    def train(self, epochs):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        test_acc_lists = []
        self.epochs = epochs
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass==self.task_size:
                     print(1)
                     opt = optim.SGD(self.model.parameters(), lr=1.0/5, weight_decay=0.00001)
                else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 5
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 62:
                if self.numclass>self.task_size:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 25
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                else:
                     opt = optim.SGD(self.model.parameters(), lr=1.0/25, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 80:
                  if self.numclass==self.task_size:
                     opt = optim.SGD(self.model.parameters(), lr=1.0 / 125,weight_decay=0.00001)
                  else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 125
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                  print("change learning rate:%.3f" % (self.learning_rate / 100))
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                #output = self.model(images)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            accuracy = self._test(self.test_loader, 1)
            print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
            torch.cuda.empty_cache()
        return accuracy

    def _test(self, testloader, mode):
        if mode==0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        corr_list = [0. for i in range(8)]
        count_list = [0. for i in range(8)]
        acc_list = []
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
            for i in np.unique(labels.squeeze().detach().cpu().numpy()):
                print(f"Label: {i}")
                mask = labels.squeeze().detach().cpu().numpy() == i
                preds = predicts[mask] if mode == 1 else outputs[mask]
                corr_list[i] += (preds.cpu().numpy() == labels[mask].cpu().numpy()).sum()
                count_list[i] += mask.sum()
        accuracy = 100 * correct / total
        for i, j in zip(corr_list, count_list):
            if j > 0:
                entry = i/j
                acc_list.append(entry.item())
            else:
                acc_list.append(0.)
        self.model.train()
        self.acc_lists.append(acc_list)
        print(f"Acc_list: {self.acc_lists}")
        #######################################################################
        storage_size = self.exemplar_set.__sizeof__()
        print(f"Number of exemplars: {len(self.exemplar_set)}")
        #######################################################################
        wandb.log({
            "test_acc":accuracy/100,
            "storage_size":storage_size,
            "acc_lists":self.acc_lists})
        return accuracy

    def _compute_loss(self, indexs, imgs, target):
        output=self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            #old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_target=torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)

    # change the size of examplar
    def afterTrain(self, accuracy):
        self.model.eval()
        m=int(self.memory_size/self.numclass)
        self._reduce_exemplar_sets(m)

        for i in range(self.numclass - self.task_size, self.numclass):
            images = None
            print('construct class %s examplar:'%(i),end='')
            for step, (indexs, data, label) in enumerate(self.train_loader):
                if images is None:
                    images = data[label.squeeze() == i]
                else:
                    images = torch.cat((images, data[label.squeeze() == i]))
                # if step == 1:
                #     break
            self._construct_exemplar_set(images,m)
        self.numclass += 1
        self.compute_exemplar_class_mean()
        self.model.train()

        KNN_accuracy=self._test(self.test_loader,0)
        print("NMS accuracy："+str(KNN_accuracy.item()))
        filename='icarl_accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, i + 10)
        filename = self.path + "_" + self.run_name + "_" + filename
        torch.save(self.model, filename)

        with open(self.acc_lists_file, "w") as f:
            print(self.acc_lists, file = f)

        self.old_model=torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()

    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
     
        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        #self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(images[0].detach()).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(images[index].detach()).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        self.model.eval()
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s"%(str(index)))
            exemplar = self.exemplar_set[index]
            #exemplar=self.train_dataset.get_image_class(index)
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_, _ = self.compute_class_mean(exemplar,self.classify_transform)
            class_mean = (class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        #test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)