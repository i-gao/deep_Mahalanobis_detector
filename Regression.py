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
from sklearn.linear_model import LogisticRegressionCV
import os

parser = argparse.ArgumentParser()
parser.add_argument('--in_data', required=True,
                    help='cifar10 | cifar100 | svhn')
parser.add_argument('--train_data', required=True,
                    help='out_dataset to train logreg on | adversarial | out')
parser.add_argument('--out_data', required=True,
                    help='all | svhn | imagenet_resize | lsun_resize | fgsm | bim')
parser.add_argument('--test_noise', type=float, default=0.01,
                    metavar='eta', help='noise magnitude added to test inputs')
parser.add_argument('--score_path', default='./output/scores/', help='path where npy score files are saved')
parser.add_argument('--verbose', type=bool, default=True, help='verbosity')
args = parser.parse_args()
print(args)

ADVERSARIAL = ["fgsm", "deepfool", "bim", "cwl2"]
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
    SAVE_PATH = './output/regression/resnet_' + args.in_data + '/' 
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    engine = MahalanobisRegression(args.in_data, SAVE_PATH)

    if args.train_data == "adversarial":
        engine.train(*ADVERSARIAL)
    elif args.train_data == "out":
        engine.train("svhn", "imagenet_resize", "lsun_resize")
    else:
        engine.train(args.train_data)

    if args.out_data == "all":
        for out_data in ["svhn", "imagenet_resize", "lsun_resize"]:
            engine.eval(out_data)
        for out_data in ADVERSARIAL:
            engine.eval(out_data)
        
        engine.eval(*ADVERSARIAL)
        engine.eval("svhn", "imagenet_resize", "lsun_resize", *ADVERSARIAL)
    else:
        engine.eval(args.out_data)

    print(engine.tnr_at_tpr95)
    print(engine.auroc)

class MahalanobisRegression:
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

        # model
        self.model = None
        self.val_data = None # out set that the model was trained on

        # return values
        self.tnr_at_tpr95 = {}
        self.auroc = {}

    def train(self, *val_data):
        """
        Trains a logistic regression to find weights used to combine layers for a simple
        threshold based detector.
        Args:
        - val_data: validation out_dataset to train logreg on
            if more than one arg provided, unions the datasets
        """
        if args.verbose:
            print(">> Training logistic regression on in_data " + self.in_data + " with out set " + str(val_data))

        # load out_data and union if necessary
        scores = self.in_file[:, :-1]
        y = self.in_file[:, -1]
        for data in val_data:
            val_file = np.load(self.load_path + 'Mahalanobis_{}_{}_{}.npy'.format(self.test_noise, self.in_data, data))
            scores = np.vstack((scores, val_file[:, :-1]))
            y = np.concatenate((y, val_file[:, -1]))
        
        self.model = LogisticRegressionCV(n_jobs=-1).fit(scores, y)
        self.val_data = val_data

    def eval(self, *out_data):
        """
        Computes TNR @ TPR95, AUROC for scores generated on the (in_data, out_data) pair
        by training a logistic regression ensemble using val_data.
        Args:
        - out_data: name of out_dataset to evaluate regression on
            if more than one arg provided, unions the datasets
        Output:
        - saves ROC curves fo in save_path with format roc_TESTNOISE_INDATA_OUTDATA_LAYERINDEX
        - saves TNR @ TPR95, AUROC scores in class variables for later use
        - saves bar graph that shows auroc varying with layer index in save_path with format bar_TESTNOISE_INDATA_OUTDATA
        """
        if self.model is None:
            raise Exception("The model has not been trained. Run train().")

        if args.verbose:
            print(">> Evaluating trained ensemble on out-dataset " + str(out_data))

        # load out_data and union if necessary
        scores = self.in_file[:, :-1]
        y = self.in_file[:, -1]
        for data in out_data:
            out_file = np.load(self.load_path + 'Mahalanobis_{}_{}_{}.npy'.format(self.test_noise, self.in_data, data))
            scores = np.vstack((scores, out_file[:, :-1]))
            y = np.concatenate((y, out_file[:, -1]))

        # run model on out_data
        y_pred = self.model.predict_proba(scores)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y, y_pred, pos_label=1)
        tnr = 1-fpr

        plt.figure()
        plt.plot(PLOT_X, -PLOT_X+1)
        plt.plot(tnr, tpr)
        plt.title("In: {} Out: {} Train: {} Ensemble".format(self.in_data, out_data, self.val_data))
        plt.ylabel("TPR")
        plt.xlabel("TNR")
        plt.savefig(self.save_path + 'roc_{}_{}_{}_{}_ensemble.png'.format(self.test_noise, self.in_data, self.val_data, out_data))

        tpr95_pos_idx = np.abs(tpr - .95).argmin()
        self.tnr_at_tpr95[out_data] = tnr[tpr95_pos_idx]
        self.auroc[out_data] = metrics.roc_auc_score(y, y_pred)

        if args.verbose:
            print("TNR @ TPR95:", self.tnr_at_tpr95[out_data])
            print("AUROC: ", self.auroc[out_data])

if __name__ == '__main__':
    main()