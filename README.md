# Experiments on CIFAR datasets with PyTorch

## Introduction
Reimplement state-of-the-art CNN models in cifar dataset with PyTorch, now including:

1.[ResNet](https://arxiv.org/abs/1512.03385v1)

2.[PreActResNet](https://arxiv.org/abs/1603.05027v3)

3.[WideResNet](https://arxiv.org/abs/1605.07146v4)

4.[ResNeXt](https://arxiv.org/abs/1611.05431v2)

5.[DenseNet](https://arxiv.org/abs/1608.06993v4)

other results will be added later.

## Requirements:software
Requirements for [PyTorch](http://pytorch.org/)

## Requirements:hardware
For most experiments, one or two K40(~11G of memory) gpus is enough cause PyTorch is very memory efficient. However,
to train DenseNet on cifar(10 or 100), you need at least 4 K40 gpus.

## Usage
1. Clone this repository

```
git clone https://github.com/junyuseu/pytorch-cifar-models.git
```

In this project, the network structure is defined in the models folder, the script ```gen_mean_std.py``` is used to calculate
the mean and standard deviation value of the dataset.

2. Edit main.py and run.sh

In the ```main.py```, you can specify the network you want to train(for example):

```
model = resnet20_cifar(num_classes=10)
...
fdir = 'result/resnet20_cifar10'
```

Then, you need specify some parameter for training in ```run.sh```. For resnet20:

```
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct 10
```

3. Train

```
nohup sh run.sh > resnet20_cifar10.log &
```

After training, the training log will be recorded in the .log file, the best model(on the test set) 
will be stored in the fdir.

**Note**:For first training, cifar10 or cifar100 dataset will be downloaded, so make sure your comuter is online.
Otherwise, download the datasets and decompress them and put them in the ```data``` folder.

4. Test

```
CUDA_VISIBLE_DEVICES=0 python main.py -e --resume=fdir/model_best.pth.tar
```

5. CIFAR100

The default setting in the code is for cifar10, to train with cifar100, you need specify it explicitly in the code.

```
model = resnet20_cifar(num_classes=100)
```

**Note**: you should also change **fdir** In the run.sh, you should set ```-ct 100```

## Results
**Note**:The results as follow are got by only one single experiment.

**We got comparable or even better results than the original papers, the experiment settings are totally follow 
the original ones**

### ResNet

layers|#params|error(%)
:---:|:---:|:---:
20|0.27M|8.33
32|0.46M|7.36
44|0.66M|6.77
56|0.85M|6.73
110|1.7M|**6.13**
1202|19.4M|-

### PreActResNet

dataset|network|baseline unit|pre-activation unit
:---:|:---:|:---:|:---:
CIFAR-10|ResNet-110|6.13|6.13
CIFAR-10|ResNet-164|5.84|5.35
CIFAR-10|ResNet-1001|11.27|**5.13**
CIFAR-100|ResNet-164|24.99|24.50
CIFAR-100|ResNet-1001|31.73|**24.03**

### WideResNet

depth-k|#params|CIFAR-10|CIFAR-100
:---:|:---:|:---:|:---:
20-10|26.8M|4.27|19.73
26-10|36.5M|**3.89**|**19.51**

### ResNeXt

network|#params|CIFAR-10|CIFAR-100
:---:|:---:|:---:|:---:
ResNeXt-29,1x64d|4.9M|4.51|22.09
ResNeXt-29,8x64d|34.4M|3.78|17.44
ResNeXt-29,16x64d|68.1M|**3.69**|**17.11**

### DenseNet

network|depth|#params|CIFAR-10|CIFAR-100
:---:|:---:|:---:|:---:|:---:
DenseNet-BC(k=12)|100|0.8M|4.69|22.19
DenseNet-BC(k=24)|250|15.3M|3.44|**17.17**
DenseNet-BC(k=40)|190|25.6M|**3.41**|17.33

# References:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[3] S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.

[4] S. Xie, G. Ross, P. Dollar, Z. Tu and K. He Aggregated residual transformations for deep neural networks. In CVPR, 2017

[5] H. Gao, Z. Liu, L. Maaten and K. Weinberger. Densely connected convolutional networks. In CVPR, 2017
