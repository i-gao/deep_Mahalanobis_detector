"""
Adapted 2020 by Irena Gao 
Forked from Kimin Lee: https://github.com/pokaxpoka/deep_Mahalanobis_detector

Functions to perturb datasets used in experiments. 
Model: resnet

"""
import argparse
import os
import torch
import torch.nn as nn
import data_loader
import numpy as np
import models
import lib.adversary as adversary
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,
                    help='cifar10 | cifar100 | svhn')
parser.add_argument('--attack', required=True,
                    help='fgsm | deepfool | bim | cwl2')
parser.add_argument('--batch_size', type=int, default=200,
                    metavar='N', help='batch size for data loader')
parser.add_argument('--data_path', default='./data', help='data path')
parser.add_argument('--verbose', type=bool, default=True, help='verbosity')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)

# constants
torch.cuda.set_device(args.gpu) 
NET_PATH = './pre_trained/resnet_' + args.dataset + '.pth'
SAVE_PATH = './output/adversarial/resnet_' + args.dataset + '/'
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

ADV_NOISE = {
    'fgsm': 0.05,
    'bim': 0.01,
    'deepfool': {
        'cifar10': 0.18,
        'cifar100': 0.03, 
        'svhn': 0.1
    },
    'cwl2': None
}
RANDOM_NOISE_SIZE = {
    'fgsm': {
        'cifar10': 0.25 / 4,
        'cifar100': 0.25 / 8, 
        'svhn': 0.24 / 4
    },
    'bim': {
        'cifar10': 0.13 / 2,
        'cifar100': 0.13 / 4, 
        'svhn': 0.13 / 2
    },
    'deepfool': {
        'cifar10': 0.25 / 4,
        'cifar100': 0.13 / 4, 
        'svhn': 0.126
    },
    'cwl2': {
        'cifar10': 0.05 / 2,
        'cifar100': 0.05 / 2, 
        'svhn': 0.05 / 1
    }
}
MIN_PIXEL = -2.42906570435
MAX_PIXEL = 2.75373125076

def main():
    """
    Args:
    - dataset: name of dataset to attack (expected cifar10/cifar100/svhn)
    - attack: type of attack to launch (expected fgsm | deepfool | bim | cwl2)
    - batch_size: batch size for data loader 
    - gpu: gpu index
    """
    # load data
    num_classes = 100 if args.dataset == 'cifar100' else 10
    _, loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.data_path)

    # load model
    model = models.ResNet34(num_c=num_classes)
    model.load_state_dict(torch.load(NET_PATH, map_location = "cuda:" + str(args.gpu)))
    model.cuda()

    # apply attack
    applyAttack(args.attack, model, loader, num_classes)

def applyAttack(attack, model, data_loader, num_classes):
    if args.verbose:
        print(">> Applying attack ", attack, " on dataset ", args.dataset)

    ## SETUP ##
    model.eval()

    adv_noise = ADV_NOISE[attack]
    random_noise_size = RANDOM_NOISE_SIZE[attack][args.dataset]

    adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
    label_tot = 0
    correct, adv_correct, noise_correct = 0, 0, 0
    total, generated_noise = 0, 0

    criterion = nn.CrossEntropyLoss().cuda()

    selected_list = []
    selected_index = 0
    
    ## ITERATE OVER DATA POINTS ##
    for data, target in data_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()

        noisy_data = torch.add(data.data, random_noise_size, torch.randn(data.size()).cuda()) 
        noisy_data = torch.clamp(noisy_data, MIN_PIXEL, MAX_PIXEL)

        if total == 0:
            clean_data_tot = data.clone().data.cpu()
            label_tot = target.clone().data.cpu()
            noisy_data_tot = noisy_data.clone().cpu()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
            label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)
            
        # generate adversarial
        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        if attack == 'fgsm': 
            gradient = adversary.fgsm(inputs)
            adv_data = torch.add(inputs.data, adv_noise, gradient)
        elif attack == 'bim': 
            gradient = adversary.bim(inputs, target, model, criterion, adv_noise, MIN_PIXEL, MAX_PIXEL) 
            adv_data = torch.add(inputs.data, adv_noise, gradient)
        if attack == 'deepfool':
            _, adv_data = adversary.deepfool(model, data.data.clone(), target.data.cpu(), \
                                             num_classes, step_size=adv_noise, train_mode=False)
            adv_data = adv_data.cuda()
        elif attack == 'cwl2':
            _, adv_data = adversary.cw(model, data.data.clone(), target.data.cpu(), 1.0, 'l2', crop_frac=1.0)

        adv_data = torch.clamp(adv_data, MIN_PIXEL, MAX_PIXEL)
        
        # measure the noise 
        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)

        if total == 0:
            adv_data_tot = adv_data.clone().cpu()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)

        output = model(Variable(adv_data, volatile=True))

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_adv = pred.eq(target.data).cpu()
        adv_correct += equal_flag_adv.sum()
        
        output = model(Variable(noisy_data, volatile=True))
        
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_noise = pred.eq(target.data).cpu()
        noise_correct += equal_flag_noise.sum()
        
        for i in range(data.size(0)):
            if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)
            selected_index += 1
            
        total += data.size(0)

    ## OUTPUT ##
    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)

    torch.save(clean_data_tot, '%s/clean_data_resnet_%s_%s.pth' % (SAVE_PATH, args.dataset, attack))
    torch.save(adv_data_tot, '%s/adv_data_resnet_%s_%s.pth' % (SAVE_PATH, args.dataset, attack))
    torch.save(noisy_data_tot, '%s/noisy_data_resnet_%s_%s.pth' % (SAVE_PATH, args.dataset, attack))
    torch.save(label_tot, '%s/label_resnet_%s_%s.pth' % (SAVE_PATH, args.dataset, attack))

    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
    print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))

if __name__ == '__main__':
    main()