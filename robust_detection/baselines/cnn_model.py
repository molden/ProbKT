import pytorch_lightning as pl
import torch
import torch.nn as nn
from argparse import ArgumentParser
from robust_detection.utils import str2bool

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet18, resnet50
from robust_detection.utils import str2bool


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=2)

        self.dropout = nn.Dropout2d(p=0.5,inplace=True)

        #print "block.expansion=",block.expansion
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        #print "avepool: ",x.data.shape
        x = x.view(x.size(0), -1)
        #print "view: ",x.data.shape

        return x


class out_layer_cnn(nn.Module):
    def __init__(self,in_features, hidden_dim, num_classes,agg_case, bypass = False):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,num_classes))
        self.agg_case = agg_case
        self.bypass = bypass
        if self.bypass:
            self.out_ = nn.Linear(num_classes,1)
    
    def forward(self,x):
        if self.agg_case:
            if self.bypass:
                class_out = self.out(x)
                sum_out = self.out_(self.out(x))
            else:
                class_out = self.out(x)
                sum_out = torch.matmul(class_out,torch.arange(10,device = x.device)[:,None].float())
        else:
            class_out = self.out(x)
            sum_out = None
        return class_out, sum_out

class CNN(pl.LightningModule):
    def __init__(self, hidden_dim,  lr, weight_decay, num_classes, input_channels = 1, pre_trained = False, agg_case = False, bypass = False,  **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.agg_case = agg_case
        self.bypass = bypass
        if pre_trained:
            resnet = resnet50(pretrained = True, progress = True)
            #out = nn.Sequential(nn.Linear(resnet.fc.in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,num_classes))
            out = out_layer_cnn(resnet.fc.in_features, hidden_dim, num_classes, agg_case, bypass = bypass)
            resnet.fc = out
            self.model = nn.Sequential(nn.Conv2d(1, 3, 1),resnet)
        else:
            resnet = resnet18(pretrained = False, progress = True)
            #out = nn.Sequential(nn.Linear(resnet.fc.in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,num_classes))
            out = out_layer_cnn(resnet.fc.in_features, hidden_dim, num_classes, agg_case, bypass = bypass)
            resnet.fc = out
            self.model = nn.Sequential(nn.Conv2d(1, 3, 1),resnet)

            #self.resnet = ResNet(block = BasicBlock, layers = [2,2,2,2], input_channels =1)
            #self.out = nn.Sequential(nn.Linear(BasicBlock.expansion * 512 * 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,num_classes))
            #self.model = torch.nn.Sequential(self.resnet,self.out)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.weight_decay)
    
    def forward(self,X):
        pred = self.model(X)
        #embed = self.resnet(X)
        #pred = self.out(embed)
        return pred

    def training_step(self, batch, batch_idx):
        X,y,agg_y, targets, og_labels = batch
        y_hat, y_agg_hat = self(X) 
        mask = torch.ones_like(y_hat)

        loss = self.compute_mse(y_hat,y,mask, y_agg_hat, agg_y, og_labels)
        
        accuracy = self.compute_accuracy(y,y_hat,mask)
        self.log("train_loss", loss,on_epoch = True)
        self.log("train_acc", accuracy,on_epoch = True)
        return {"loss": loss}

    def compute_mse(self,pred,target,mask, y_agg_hat, agg_y, og_labels):
        if y_agg_hat is not None:
            if self.bypass:
                return ((y_agg_hat-agg_y).pow(2)).mean()
            else:
                return ((y_agg_hat-agg_y).pow(2)).mean() + (((pred-target).pow(2)*mask) * og_labels).sum() / (0.0001 + (mask * og_labels).sum())
        else:
            return ((pred-target).pow(2)*mask).sum()/mask.sum()
    
    def compute_accuracy(self,y,y_pred,mask):
        return ((y_pred.round()*mask)==(y*mask)).all(1).float().mean()
        
    def validation_step(self, batch, batch_idx):
        X,y, agg_y, targets, og_labels = batch
        y_hat, y_agg_hat = self(X) 
        mask = torch.ones_like(y_hat)
        loss = self.compute_mse(y_hat,y,mask, y_agg_hat, agg_y, og_labels)
        accuracy = self.compute_accuracy(y,y_hat,mask)
        self.log("val_loss", loss,on_epoch = True)
        self.log("val_acc", accuracy,on_epoch = True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        X,y, agg_y, targets, og_labels = batch
        y_hat, y_agg_hat = self(X) 
        mask = torch.ones_like(y_hat)

        loss = self.compute_mse(y_hat,y,mask, y_agg_hat, agg_y, og_labels)
        accuracy = self.compute_accuracy(y,y_hat,mask)
        self.log("test_loss", loss,on_epoch = True)
        self.log("test_acc", accuracy,on_epoch = True)
        return {"loss": loss, "Y_pred": y_hat, "Y":y, "M": mask}

    def predict_step(self, batch, batch_idx):
        X,y, agg_y, targets, og_labels = batch
        y_hat, y_agg_hat = self(X) 
        mask = torch.ones_like(y_hat)

        loss = self.compute_mse(y_hat,y,mask, y_agg_hat, agg_y, og_labels)
        accuracy = self.compute_accuracy(y,y_hat,mask)
        return {"loss": loss, "Y_pred":y_hat, "Y":y, "M": mask, "boxes_true":[t["boxes"] for t in targets], "targets":[t["labels"] for t in targets], "X":X, "og_labels":og_labels, "agg_y":agg_y, "y_agg_hat":y_agg_hat}
        
    @classmethod
    def add_model_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help = False)
        parser.add_argument('--hidden_dim', type=int, default = 128)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.add_argument('--pre_trained', type=str2bool, default=False)
        parser.add_argument('--agg_case', type=str2bool, default=False)
        parser.add_argument('--bypass', type=str2bool, default=False, help= "if true, computes directly the agg quantity. Only to be used with agg_case = True")
        return parser
