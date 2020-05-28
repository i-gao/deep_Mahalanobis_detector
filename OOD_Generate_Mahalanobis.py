"""
Adapted 2020 by Irena Gao 
Forked from Kimin Lee: https://github.com/pokaxpoka/deep_Mahalanobis_detector

Generates Mahalanobis scores for an (in-dataset, out-dataset) pair. 
Model: DenseNet3

"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import models
import os
import lib_generation

from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--in_data', required=True,
                    help='cifar10 | cifar100 | svhn')
parser.add_argument('--out_data', required=True,
                    help='cifar10 | cifar100 | svhn')
parser.add_argument('--batch_size', type=int, default=200,
                    metavar='N', help='batch size for data loader')
parser.add_argument('--test_noise', type=float, default=0.01,
                    metavar='eta', help='batch size for data loader')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)

torch.cuda.manual_seed(0)
torch.cuda.set_device(args.gpu)

### MODIFYING ./output/ -> ./output/scores/ ####


def main():
    """
    Args:
    - in_data: name of in-dataset
    - out_data: name of out-dataset
    - batch_size: batch size for data loader (both in/out)
    - test noise: constant noise args.test_noise applied to test inputs for separation (eta/epsilon)
    - gpu: gpu index
    """
    ### SETUP ###
    # process args
    NET_PATH = './pre_trained/densenet_' + args.in_data + '.pth'
    OUT_PATH = './output/scores/densenet_' + args.in_data + '/'
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)

    NUM_CLASSES = 100 if args.in_data == 'cifar100' else 10

    # load model
    if args.in_data == 'svhn':
        model = models.DenseNet3(100, int(NUM_CLASSES))
        model.load_state_dict(torch.load(
            NET_PATH, map_location="cuda:" + str(args.gpu)))
    else:
        model = torch.load(NET_PATH, map_location="cuda:" + str(args.gpu))
    model.cuda()

    # load dataset
    train_loader, in_test_loader = data_loader.getTargetDataSet(
        args.in_data, args.batch_size)
    out_test_loader = data_loader.getNonTargetDataSet(
        args.out_data, args.batch_size)

    # get information about num layers, activation sizes by trying a test input
    model.eval()
    temp_x = Variable(torch.rand(2, 3, 32, 32).cuda())
    temp_list = model.feature_list(temp_x)[1]

    NUM_LAYERS = len(temp_list)
    ACTIVATION_SIZES = [layer.size(1) for layer in temp_list]

    ### TRAINING ###
    sample_mean, precision = lib_generation.sample_estimator(
        model, NUM_CLASSES, ACTIVATION_SIZES, train_loader)

    ### TESTING ###
    # test inliers
    for i in range(NUM_LAYERS):
        M_in = lib_generation.get_Mahalanobis_score(model, in_test_loader, NUM_CLASSES, OUT_PATH,
                                                    True, sample_mean, precision, i, args.test_noise)
        M_in = np.asarray(M_in, dtype=np.float32)
        if i == 0:
            Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
        else:
            Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
    
    # test outliers
    for i in range(NUM_LAYERS):
        M_out = lib_generation.get_Mahalanobis_score(model, out_test_loader, NUM_CLASSES, OUT_PATH,
                                                        False, sample_mean, precision, i, args.test_noise)
        M_out = np.asarray(M_out, dtype=np.float32)
        if i == 0:
            Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
        else:
            Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

    # save results
    Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
    Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)

    Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(
        Mahalanobis_out, Mahalanobis_in)
    Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
    
    file_name = os.path.join(OUT_PATH, 'Mahalanobis_%s_%s_%s.npy' % (str(args.test_noise), args.in_data, args.out_data))
    np.save(file_name, Mahalanobis_data)

if __name__ == '__main__':
    main()
