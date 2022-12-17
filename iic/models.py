import torch
import torch.nn as nn
import torch.nn.functional as F

from sys import float_info
import pytorch_lightning as pl

from utils.data_preprocessing import recombine_image

EPS = float_info.epsilon


def IIC_Loss(y1, y2, inverse_transform, lamb=1.0, padding=10):
    bn, k, h, w = y1.shape

    y2_ = inverse_transform.backward_transform(y2)
    y2_ = y2_.permute(1, 0, 2, 3)
    y1 = y1.permute(1, 0, 2, 3)

    p_i_j = F.conv2d(y1, weight=y2_, padding=padding)
    p_i_j = p_i_j.permute(2, 3, 0, 1)
    p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
    p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0

    p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1, 1, k, 1)
    p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1, 1, 1, k)

    # for log stability; tiny values cancelled out by mult with p_i_j anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_mat[(p_i_mat < EPS).data] = EPS
    p_j_mat[(p_j_mat < EPS).data] = EPS
    loss = (-p_i_j * torch.log(p_i_j / (p_i_mat * p_j_mat)))
    return loss.sum(axis=0).sum()


def convolution_segment(in_feat, out_feat, kernel=2, padding=2, dilation=1,
                        batch_norm=True, max_pool=False, activation=True):
    layers = []
    layers.append(nn.Conv2d(in_feat, out_feat, kernel, padding=padding, dilation=dilation, bias=False))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_feat))

    if max_pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
    if activation:
        layers.append(nn.ReLU())
    return layers


class AbstractVGGTrunk(nn.Module):
    def __init__(self, input_channels, nr_classes, input_size):
        super().__init__()
        self.input_channels = input_channels
        self.nr_classes = nr_classes
        self.input_size = input_size


class VGGTrunk(AbstractVGGTrunk):
    def __init__(self, input_channels, nr_classes, input_size=16, verbose=False):
        super().__init__(input_channels, nr_classes, input_size)
        self.convolution = nn.Sequential(
            *convolution_segment(input_channels, 32, kernel=3, padding=0),
            *convolution_segment(32, 32, kernel=3, padding=0),
            *convolution_segment(32, 64, kernel=1, dilation=1, padding=0, activation=True, batch_norm=False,
                                 max_pool=False),
        )
        self.convolution_1 = nn.Sequential(
            *convolution_segment(input_channels, 32, kernel=1, padding=0),
            *convolution_segment(32, 64, kernel=1, dilation=1, padding=0, activation=True, batch_norm=False,
                                 max_pool=False),
        )

        self.convolve_output = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, dilation=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, nr_classes, kernel_size=1, stride=1, dilation=1, padding=0, bias=False),
            nn.Softmax2d()
        )
        self.double()
        self.verbose = verbose

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x_0 = self.convolution(x)
        x_1 = self.convolution_1(x)
        if self.verbose:
            print("Intermediate Size: ", x_0.shape, x_1.shape)
        x_0 = F.interpolate(x_0, size=self.input_size, mode="bilinear")
        out = x_0 + x_1
        return self.convolve_output(out)


class IICModel(pl.LightningModule):
    def __init__(self, net: AbstractVGGTrunk, loss_padding=10,
                 val_sample=None, recombination_size=None, sub_image_size=None, crop_factor=None):
        super().__init__()
        self.net = net
        self.net._initialize_weights()
        self.nr_classes = net.nr_classes
        self.loss_padding = loss_padding

        self.step = 0
        self.val_sample = val_sample
        self.recombination_size = recombination_size
        self.sub_image_size = sub_image_size
        self.crop_factor = crop_factor

    def forward(self, x):
        return self.net(x)

    def train_evaluation(self, train_batch):
        img1 = train_batch["img1"]
        img2 = train_batch["img2"]


        y1 = self(img1)
        y2 = self(img2)

        loss = IIC_Loss(y1, y2, train_batch["inverse"], padding=self.loss_padding)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        return [optimizer]  # , [scheduler]

    def training_step(self, train_batch, batch_idx):
        loss = self.train_evaluation(train_batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        # self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, train_batch, batch_idx):
        if self.val_sample is None:
            loss = self.train_evaluation(train_batch)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            return loss
        else:
            sample_clf = self.net(self.val_sample).detach().cpu()
            sample_clf = recombine_image(sample_clf, self.recombination_size,
                                         int(self.sub_image_size * self.crop_factor))
            self.step += 1
            for ch in range(self.nr_classes):
                self.logger.experiment.add_image(f'generated_map_{ch}', sample_clf[ch, :, :], self.step,
                                                 dataformats="HW")
            # self.log("val_acc", acc, prog_bar=True, on_epoch=True)
            return 0.0
