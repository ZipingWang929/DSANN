import torch.nn as nn
from torchvision import models
import torch
from torch.autograd import Function
from collections import OrderedDict
import torch.nn.functional as F

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()


class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier" + str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features
        self.bottleneck = nn.Sequential(nn.Linear(model_alexnet.classifier[6].in_features, 512),
                                        nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.bottleneck(x)

        return x

    def output_num(self):
        #         return self._feature_dim

        return 512


class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet

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
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim


class GeneDistrNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, latent_size):
        super(GeneDistrNet, self).__init__()
        self.num_labels = num_classes
        self.latent_size = latent_size
        self.genedistri = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(input_size + self.num_labels, self.latent_size)),
            ("relu1", nn.LeakyReLU()),

            ("fc2", nn.Linear(self.latent_size, hidden_size)),
            ("relu2", nn.ReLU()),
        ]))
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            layer.apply(init_weights)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = x.cuda()
        x = self.genedistri(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_labels, rp_size ):
        super(Discriminator, self).__init__()
        self.features_pro = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, y, z):
        feature = z.view(z.size(0), -1)
        logit = self.features_pro(feature)
        return logit


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class feat_classifier(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier, self).__init__()
        #         self.type = type
        #         self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        #         self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(input_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        # x = self.fc0(x)
        x = self.fc1(x)
        return x


#
class DSANN(nn.Module):
    def __init__(self, base_name, num_domains, num_classes):
        super(DSANN, self).__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.base_network = get_backbone(base_name)
        self.input_size = self.base_network.output_num()
        self.classifiers = nn.ModuleList(
            [feat_classifier(self.num_classes, self.input_size) for i in range(num_domains)])
        self.Discriminators = nn.ModuleList(
            [Discriminator(self.input_size, num_classes, 128) for i in range(num_domains)])

    def forward(self, x, y=None, mark=None, train=True):
        if train:
            feature = self.base_network(x)
            cls = self.classifiers[mark](feature)
            y_onehot = torch.zeros(y.size(0), self.num_classes).cuda()
            y_onehot.scatter_(1, y.view(-1, 1), 0.7).cuda()
            dis = self.Discriminators[mark](y_onehot, feature)
            return cls, dis
        else:
            cls = []
            dis = []
            for i in range(self.num_domains):
                feature = self.base_network(x)
                cls_ = self.classifiers[i](feature)
                y = cls_.argmax(1)
                y_onehot = torch.zeros(y.size(0), self.num_classes).cuda()
                y_onehot.scatter_(1, y.view(-1, 1), 0.7).cuda()
                cls.append(F.softmax(cls_))
                dis.append(self.Discriminators[i](y_onehot, feature))
            return cls, dis


    def get_parameters1(self, initial_lr=1.0):
        params = [{'params': self.base_network.parameters(), 'lr': 1.0 * initial_lr}]
        params.append({'params': self.classifiers.parameters(), 'lr': 1.0 * initial_lr})
        params.append({'params': self.Discriminators.parameters(), 'lr': 1.0 * initial_lr})
        return params