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

torch.cuda.manual_seed(0)

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
    torch.cuda.set_device(args.gpu) 
    NET_PATH = './pre_trained/densenet_' + args.in_data + '.pth'
    SAVE_PATH = './output/scores/densenet_' + args.in_data + '/'
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    engine = MahalanobisGenerator(args.in_data, NET_PATH, SAVE_PATH)
    engine.train()
    engine.test(args.out_data)


class MahalanobisGenerator:
    """
    Class to generate Mahalonobis scores for ONE fixed training set.
    """

    def __init__(self, in_data, net_path, save_path, batch_size=args.batch_size):
        """
        Args:
        - in_data: name of in-dataset
        - net_path: path to pre-trained net weights (corresponding to in_data)
        - save_path: output file path, must exist
        - batch_size: batch size for data loader (both in/out)
        """
        self.in_data = in_data
        self.num_classes = 100 if self.in_data == 'cifar100' else 10
        self.batch_size = batch_size
        self.save_path = save_path

        # load training data
        self.train_loader = data_loader.getTargetDataSet(
            self.in_data, self.batch_size)

        # load model
        if self.in_data == 'svhn':
            self.model = models.DenseNet3(100, self.num_classes)
            self.model.load_state_dict(torch.load(
                net_path, map_location="cuda:" + str(args.gpu)))
        else:
            self.model = torch.load(
                net_path, map_location="cuda:" + str(args.gpu))
        self.model.cuda()

        # get information about num layers, activation sizes by trying a test input
        self.model.eval()
        temp_x = Variable(torch.rand(2, 3, 32, 32).cuda())
        temp_list = self.model.feature_list(temp_x)[1]

        self.num_layers = len(temp_list)
        self.activation_sizes = [layer.size(1) for layer in temp_list]

    ### TRAINING ###
    def train(self):
        self.sample_mean, self.precision = lib_generation.sample_estimator(
            self.model, self.num_classes, self.activation_sizes, self.train_loader)

    ### TESTING ###
    def test(self, out_data, in_data=None, test_noise=args.test_noise):
        in_test_loader = self.train_loader if in_data is None else data_loader.getNonTargetDataSet(
            in_data, self.batch_size)

        out_test_loader = data_loader.getNonTargetDataSet(
            out_data, self.batch_size)

        # test inliers
        for i in range(self.num_layers):
            M_in = lib_generation.get_Mahalanobis_score(self.model, in_test_loader, self.num_classes, self.save_path,
                                                        True, self.sample_mean, self.precision, i, test_noise)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate(
                    (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        # test outliers
        for i in range(self.num_layers):
            M_out = lib_generation.get_Mahalanobis_score(self.model, out_test_loader, self.num_classes, self.save_path,
                                                            False, self.sample_mean, self.precision, i, test_noise)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate(
                    (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

        # save results
        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)

        Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(
            Mahalanobis_out, Mahalanobis_in)
        Mahalanobis_data = np.concatenate(
            (Mahalanobis_data, Mahalanobis_labels), axis=1)

        file_name = os.path.join(self.save_path, 'Mahalanobis_%s_%s_%s.npy' % (
            str(test_noise), self.in_data, out_data))
        np.save(file_name, Mahalanobis_data)

if __name__ == '__main__':
    main()