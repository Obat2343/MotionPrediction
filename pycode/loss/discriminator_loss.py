import sys
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

############### DISCRIMINATOR #####################
class Standard_GAN_Loss(nn.Module):
    def __init__(self, cfg, device):
        super(Standard_GAN_Loss, self).__init__()
        self.real_label = torch.ones(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.fake_label = torch.zeros(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.BCE = nn.BCEWithLogitsLoss().to(device)
        self.device = device
        self.weight = cfg.LOSS.GAN.WEIGHT

    def forward(self, fake_prediction, real_prediction):
        fake_prediction = fake_prediction.to(self.device)
        real_prediction = real_prediction.to(self.device)

        real_loss = self.BCE(real_prediction, self.real_label)
        fake_loss = self.BCE(fake_prediction, self.fake_label)

        loss = real_loss + fake_loss
        loss_dict = {}
        loss_dict['{}/standard_gan_real'.format(mode)] = real_loss.item()
        loss_dict['{}/standard_gan_fake'.format(mode)] = fake_loss.item()
        loss_dict['{}/standard_gan_loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_gan_loss'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss_d'.format(mode)] = loss.item()

        loss_dict['weight/standard_gan'] = self.weight
        return loss, loss_dict

class Relative_GAN_Loss(nn.Module):
    def __init__(self, cfg, device):
        super(Relative_GAN_Loss, self).__init__()
        self.real_label = torch.ones(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.fake_label = torch.zeros(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.BCE = nn.BCEWithLogitsLoss().to(device)
        self.device = device
        self.weight = cfg.LOSS.GAN.WEIGHT

    def forward(self, fake_prediction, real_prediction, mode):
        fake_prediction = fake_prediction.to(self.device)
        real_prediction = real_prediction.to(self.device)

        loss = self.BCE(real_prediction - fake_prediction, self.real_label)

        loss_dict = {}
        loss_dict['{}/relative_gan_loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_relative_gan_loss'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss_d'.format(mode)] = loss.item()

        loss_dict['weight/ralative_gan'] = self.weight
        return loss, loss_dict

class Relative_Average_Loss(nn.Module):
    def __init__(self, cfg, device):
        super(Relative_Average_Loss, self).__init__()
        self.real_label = torch.ones(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.fake_label = torch.zeros(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.BCE = nn.BCEWithLogitsLoss().to(device)
        self.device = device
        self.weight = cfg.LOSS.GAN.WEIGHT

    def forward(self, fake_prediction, real_prediction, mode):
        fake_prediction = fake_prediction.to(self.device)
        real_prediction = real_prediction.to(self.device)

        real_loss = self.BCE(real_prediction - torch.mean(fake_prediction), self.real_label)
        fake_loss = self.BCE(fake_prediction - torch.mean(real_prediction), self.fake_label)

        loss = real_loss + fake_loss

        loss_dict = {}
        loss_dict['{}/relative_average_real'.format(mode)] = real_loss.item()
        loss_dict['{}/relative_average_fake'.format(mode)] = fake_loss.item()
        loss_dict['{}/relative_average_loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_relative_average_loss'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss_d'.format(mode)] = loss.item()

        loss_dict['weight/relative_average_gan'] = self.weight
        return loss, loss_dict

class Relative_Average_Hinge_Loss(nn.Module):
    def __init__(self, cfg, device):
        super(Relative_Average_Hinge_Loss, self).__init__()
        self.real_label = torch.ones(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.fake_label = torch.zeros(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.BCE = nn.BCEWithLogitsLoss().to(device)
        self.device = device
        self.weight = cfg.LOSS.GAN.WEIGHT

    def forward(self, fake_prediction, real_prediction, mode):
        fake_prediction = fake_prediction.to(self.device)
        real_prediction = real_prediction.to(self.device)

        real_loss = torch.mean(torch.nn.ReLU()(1.0 - (real_prediction - fake_prediction)))
        fake_loss = torch.mean(torch.nn.ReLU()(1.0 + (fake_prediction - real_prediction)))

        loss = real_loss + fake_loss

        loss_dict = {}
        loss_dict['{}/relative_average_real'.format(mode)] = real_loss.item()
        loss_dict['{}/relative_average_fake'.format(mode)] = fake_loss.item()
        loss_dict['{}/relative_average_loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_relative_average_loss'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss_d'.format(mode)] = loss.item()

        loss_dict['weight/relative_average_hinge_loss'] = self.weight
        return loss, loss_dict

class W_GAN_Loss(nn.Module):
    def __init__(self, cfg, device):
        super(W_GAN_Loss, self).__init__()
        self.real_label = torch.ones(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.fake_label = torch.zeros(cfg.BASIC.BATCH_SIZE, 1).to(device)
        self.BCE = nn.BCEWithLogitsLoss().to(device)
        self.device = device
        self.weight = cfg.LOSS.GAN.WEIGHT

    def forward(self, fake_prediction, real_prediction, mode):
        fake_prediction = fake_prediction.to(self.device)
        real_prediction = real_prediction.to(self.device)

        loss = -torch.mean(real_prediction) + torch.mean(fake_prediction)

        loss_dict = {}
        loss_dict['{}/wgan_loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_wgan_loss'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss_d'.format(mode)] = loss.item()

        loss_dict['weight/w_gan'] = self.weight
        return loss, loss_dict

class Gradient_Penalty(nn.Module):
    def __init__(self, cfg, device):
        super(Gradient_Penalty, self).__init__()
        self.device = device
        self.weight = cfg.LOSS.GP.WEIGHT

    def forward(self, generated_image, gt_image, discriminator, mode):
        batch_size = gt_image.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(gt_image).to(self.device)
        interpolated = alpha * gt_image.data + (1 - alpha) * generated_image.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                            create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        loss = ((gradients_norm - 1) ** 2).mean()
        
        loss_dict = {}
        loss_dict['{}/gradient_penalty'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_gradient_penalty'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss_d'.format(mode)] = loss.item()

        loss_dict['weight/gradient_penalty'] = self.weight
        return loss, loss_dict