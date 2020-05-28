"""
Adapted 2020 by Irena Gao 
Forked from Kimin Lee: https://github.com/pokaxpoka/deep_Mahalanobis_detector

Generates Mahalanobis scores for an (in-dataset, out-dataset) pair. 
Model: resnet

"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import models
import os
import lib_generation
import sklearn.covariance

from torch.autograd import Variable

torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--in_data', required=True,
                    help='cifar10 | cifar100 | svhn')
parser.add_argument('--out_data', required=True,
                    help='all | svhn | imagenet_resize | lsun_resize | fgsm | deepfool | bim | cwl2')
parser.add_argument('--batch_size', type=int, default=200,
                    metavar='N', help='batch size for data loader')
parser.add_argument('--test_noise', type=float, default=0.01,
                    metavar='eta', help='batch size for data loader')
parser.add_argument('--data_path', default='./data', help='data path')
parser.add_argument('--verbose', type=bool, default=True, help='verbosity')
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
    NET_PATH = './pre_trained/resnet_' + args.in_data + '.pth'
    SAVE_PATH = './output/scores/resnet_' + args.in_data + '/'
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    engine = MahalanobisGenerator(args.in_data, NET_PATH, SAVE_PATH)
    engine.train()

    if args.out_data == "all":
        for out_data in ["svhn", "imagenet_resize", "lsun_resize"]:
            engine.test(out_data, False)
        for out_data in ["fgsm", "deepfool", "bim", "cwl2"]:
            engine.test(out_data, True)
    else:
        engine.test(args.out_data, args.out_data in ["fgsm", "deepfool", "bim", "cwl2"])


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
        self.train_loader, self.in_test_loader = data_loader.getTargetDataSet(
            self.in_data, self.batch_size, args.data_path)

        # load model
        self.model = models.ResNet34(num_c=self.num_classes)
        self.model.load_state_dict(torch.load(net_path, map_location = "cuda:" + str(args.gpu)))
        self.model.cuda()

        # get information about num layers, activation sizes by trying a test input
        self.model.eval()
        temp_x = Variable(torch.rand(2, 3, 32, 32).cuda())
        temp_list = self.model.feature_list(temp_x)[1]

        self.num_layers = len(temp_list)
        self.activation_sizes = [layer.size(1) for layer in temp_list]

    ### TRAINING ###
    def train(self):
        """
        Memorizes sample stats for in_data
        """
        if args.verbose:
            print(">> Training on in-dataset ", self.in_data)

        self.sample_mean, self.precision = self._get_sample_stats()

    ### TESTING ###
    def test(self, out_data, adversarial, in_data=None, test_noise=args.test_noise):
        """
        Computes Mahalanobis scores for test set in_data, out_data
        Args:
        - out_data: name of out dataset
        - adversarial: boolean flag for if this is an adversarial input
        - in_data: name of in dataset, default is the class's in dataset
        - test_noise: constant noise to add to test data to help separation
        """
        if in_data is None:
            in_data = self.in_data
            in_test_loader = self.in_test_loader 
        else:
            _, in_test_loader = data_loader.getNonTargetDataSet(in_data, self.batch_size, args.data_path)

        # if not adversarial, using torch.DataLoader, so just load once
        if not adversarial:
            out_test_loader = data_loader.getNonTargetDataSet(out_data, self.batch_size, args.data_path)

        # test inliers
        if args.verbose:
            print(">> Testing on in-dataset ", self.in_data)

        for i in range(self.num_layers):
            M_in = self._get_Mahalanobis_score(in_data, in_test_loader, i, test_noise)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate(
                    (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        # test outliers
        if args.verbose:
            print(">> Testing on out-dataset ", out_data)
        for i in range(self.num_layers):
            
            # if adversarial, using a list, and for some reason, need to load this continuously
            if adversarial:
                out_test_loader = data_loader.getAdversarialDataSet(out_data, self.in_data, self.batch_size)
            
            M_out = self._get_Mahalanobis_score(out_data, out_test_loader, i, test_noise)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate(
                    (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

        # save results
        if args.verbose:
            print(">> Writing results to ", self.save_path)

        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)

        Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(
            Mahalanobis_out, Mahalanobis_in)
        Mahalanobis_data = np.concatenate(
            (Mahalanobis_data, Mahalanobis_labels), axis=1)

        file_name = os.path.join(self.save_path, 'Mahalanobis_%s_%s_%s.npy' % (
            str(test_noise), self.in_data, out_data))
        np.save(file_name, Mahalanobis_data)

    ## HELPER FUNCTIONS ##
    def _get_sample_stats(self):
        """
        Computes sample mean and precision (inverse of covariance) for self.in_data
        """
        self.model.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
    
        num_sample_per_class = np.empty(self.num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(self.num_layers):
            temp_list = []
            for j in range(self.num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        for data, target in self.train_loader:
            total += data.size(0)
            data = data.cuda()
            data = Variable(data)
            with torch.no_grad():
                output, out_features = self.model.feature_list(data)
        
            for i in range(self.num_layers):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
            
            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()
        
            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    for j, out in enumerate(out_features):
                        list_features[j][label] = out[i].view(1, -1)
                else:
                    for j, out in enumerate(out_features):
                        list_features[j][label] = torch.cat((list_features[j][label], out[i].view(1, -1)), 0)             
                num_sample_per_class[label] += 1

        print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
        
        # get means
        sample_mean = []
        for i, size in enumerate(self.activation_sizes):
            temp_list = torch.Tensor(self.num_classes, int(size)).cuda()
            for j in range(self.num_classes):
                temp_list[j] = torch.mean(list_features[i][j], 0)
            sample_mean.append(temp_list)

        # get precision 
        precision = []
        for k in range(self.num_layers):
            X = 0
            for i in range(self.num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_mean[k][i]), 0)
            # find inverse            
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().cuda()
            precision.append(temp_precision)
        
        return sample_mean, precision

    def _get_Mahalanobis_score(self, test_name, test_loader, layer_index, magnitude):
        '''
        Computes Mahalanobis scores for layer @ layer_index
        - test_name: name of test dataset
        - test_loader: test data to compute scores for
        - layer_index: layer to compute scores over
        - magnitude: test time constant noise to add

        return: 
        - Mahalanobis score from layer_index
        '''
        self.model.eval()
        Mahalanobis = []
        
        temp_file_name = '%s/confidence_Ga%s_%s.txt'%(self.save_path, str(layer_index), test_name)            
        g = open(temp_file_name, 'w')
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad = True), Variable(target)
            out_features = self.model.intermediate_forward(data, layer_index)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)
            
            # compute Mahalanobis score
            gaussian_score = 0
            for i in range(self.num_classes):
                batch_sample_mean = self.sample_mean[layer_index][i]
                zero_f = out_features.data - batch_sample_mean
                term_gau = -0.5*torch.mm(torch.mm(zero_f, self.precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1,1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
            
            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = self.sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - Variable(batch_sample_mean)
            pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(self.precision[layer_index])), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()
            
            gradient =  torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
            tempInputs = torch.add(data.data, -magnitude, gradient)
    
            noise_out_features = self.model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(self.num_classes):
                batch_sample_mean = self.sample_mean[layer_index][i]
                zero_f = noise_out_features.data - batch_sample_mean
                term_gau = -0.5*torch.mm(torch.mm(zero_f, self.precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1,1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
            Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
            
            for i in range(data.size(0)):
                g.write("{}\n".format(noise_gaussian_score[i]))
        g.close()

        return Mahalanobis

if __name__ == '__main__':
    main()