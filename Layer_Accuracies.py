"""
Written 2020 by Irena Gao 

Evaluates individual layer Mahalanobis score accuracies for an (in-dataset, out-dataset) pair
using a simple threshold detector.

"""
from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os

parser = argparse.ArgumentParser()
parser.add_argument('--in_data', required=True,
                    help='cifar10 | cifar100 | svhn')
parser.add_argument('--out_data', required=True,
                    help='all | svhn | imagenet_resize | lsun_resize | fgsm | bim')
parser.add_argument('--test_noise', type=float, default=0.01,
                    metavar='eta', help='noise magnitude added to test inputs')
parser.add_argument('--score_path', default='./output/scores/', help='path where npy score files are saved')
parser.add_argument('--verbose', type=bool, default=True, help='verbosity')
args = parser.parse_args()
print(args)

ADVERSARIAL = ["fgsm", "deepfool", "bim", "cwl2"]
OUT = ["svhn", "cifar10", "cifar100", "imagenet_resize", "lsun_resize"]
# ADVERSARIAL = ["fgsm", "bim"]
PLOT_X = np.linspace(0, 1, 100)

def main():
    """
    Args:
    - in_data: name of in-dataset
    - out_data: name of out-dataset
    - batch_size: batch size for data loader (both in/out)
    - test noise: constant noise args.test_noise applied to test inputs for separation (eta/epsilon)
    - gpu: gpu index
    """
    SAVE_PATH = './output/accuracy/resnet_' + args.in_data + '/' 
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    engine = MahalanobisEvaluator(args.in_data, SAVE_PATH)

    if args.out_data == "all":
        for out_data in OUT:
            engine.eval(out_data)
        for out_data in ADVERSARIAL:
            engine.eval(out_data)
    else:
        engine.eval(args.out_data)

    print(engine.tnr_at_tpr95)
    print(engine.auroc)

class MahalanobisEvaluator:
    """
    Class to eval accuracies for Mahalonobis scores on ONE fixed in-set.
    """
    def __init__(self, in_data, save_path, load_path=args.score_path, test_noise=args.test_noise):
        """
        Args:
        - in_data: name of in-dataset
        - save_path: output file path, must exist
        - load_path, path to generated scores, must exist
        - test_noise: noise level used to generate scores, for finding file 
        
        Generated scores are expected to have the following naming format: 
        - resnet_INDATA/Mahalanobis_TESTNOISE_INDATA_TESTDATA.npy
        """
        self.in_data = in_data
        self.test_noise = test_noise
        self.save_path = save_path
        self.load_path = load_path + 'resnet_' + in_data + '/'

        # load in-test scores
        self.in_file = np.load(self.load_path + 'Mahalanobis_{}_{}_{}.npy'.format(test_noise, in_data, in_data))

        # return values
        self.tnr_at_tpr95 = {}
        self.auroc = {}

    def eval(self, out_data):
        """
        Computes TNR @ TPR95, AUROC for scores generated on the (in_data, out_data) pair.
        Args:
        - out_data: name of out_dataset to evaluate
        Output:
        - saves ROC curves for each layer in save_path with format roc_TESTNOISE_INDATA_OUTDATA_LAYERINDEX
        - saves TNR @ TPR95, AUROC scores in class variables for later use
        - saves bar graph that shows auroc varying with layer index in save_path with format bar_TESTNOISE_INDATA_OUTDATA
        """
        if out_data == self.in_data:
            print("Cannot evaluate on in-dataset.")
            return

        out_file = np.load(self.load_path + 'Mahalanobis_{}_{}_{}.npy'.format(self.test_noise, self.in_data, out_data))
        scores = np.vstack((self.in_file[:, :-1], out_file[:, :-1]))
        y = np.concatenate((self.in_file[:, -1], out_file[:, -1]))

        # for each layer, get roc curve, AUROC, and TNR @ 95 TPR
        self.tnr_at_tpr95[out_data] = np.zeros(scores.shape[1])
        self.auroc[out_data] = np.zeros(scores.shape[1])
        for i in range(scores.shape[1]): 
            fpr, tpr, _ = metrics.roc_curve(y, scores[:,i], pos_label=1)
            tnr = 1-fpr

            plt.figure()
            plt.plot(PLOT_X, -PLOT_X+1)
            plt.plot(tnr, tpr)
            plt.title("In: {} Out: {} Layer: {}".format(self.in_data, out_data, i))
            plt.ylabel("TPR")
            plt.xlabel("TNR")
            plt.savefig(self.save_path + 'roc_{}_{}_{}_{}.png'.format(self.test_noise, self.in_data, out_data, i))

            tpr95_pos_idx = np.abs(tpr - .95).argmin()
            self.tnr_at_tpr95[out_data][i] = tnr[tpr95_pos_idx]
            self.auroc[out_data][i] = metrics.roc_auc_score(y, scores[:, i])

        # generate bar graph
        plt.figure()
        plt.bar(np.arange(scores.shape[1]), self.auroc[out_data])        
        plt.title("In: {} Out: {}".format(self.in_data, out_data))
        plt.xlabel("Layer Index")
        plt.ylabel("AUROC")
        plt.savefig(self.save_path + 'bar_{}_{}_{}.png'.format(self.test_noise, self.in_data, out_data))

if __name__ == '__main__':
    main()