# Adversarial Training for OOD Detection
Irena Gao, Ryan Han, David Yue - CS229 Spring 2020

This project studies the effect of training on adversarially-perturbed input examples on robustness when detecting out-of-distribution (OOD) examples at test-time. The detection method is based on the paper "[A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888)".

## Downloading Out-of-Distribtion Datasets
The code to download CIFAR-10, CIFAR-100, and SVHN is provided. Additionally, these OOD sets must be downloaded and placed in `./data`:

* [Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

## Downloading Pre-trained Models
We use three pre-trained neural networks: three ResNets trained on CIFAR-10, CIFAR-100 and SVHN. These models are from the authors of the original paper.

Weights must be downloaded and placed in `./pre_trained/`:

* [ResNet on CIFAR-10](https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth?dl=0) / [ResNet on CIFAR-100](https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth?dl=0) / [ResNet on SVHN](https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth?dl=0)

# Suggest Workflow

```
# model: ResNet, in-distribution: CIFAR-10, gpu: 0
python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type resnet --gpu 0
```

## Detecting Out-of-Distribution Samples (Mahalanobis detector)

### 1. Extract detection characteristics:
```
### model: ResNet, in-distribution: CIFAR-10, gpu: 0  ###

# generate adversarially perturbed in-dist data:
python Generate_Adversarial.py.py --dataset cifar10 --attack all --gpu 0

# generate Mahalanobis scores on in-dist and all out-dist data:
python Generate_Scores.py --in_data cifar10 --out_data all --gpu 0

# compute the effect of layer index on score accuracy
python Layer_Accuracies.py --in_data cifar10 --out_data all

# compute logistic regressions that combine layers and accuracy
python Regression.py --in_data cifar10 --train_data fgsm --out_data all
```
