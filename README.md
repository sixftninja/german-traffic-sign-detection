# PyTorch implementation of German Traffic Sign Classification Challenge

The German Traffic Sign Recognition Benchmark (GTSRB) is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. It consists of about 50,000 images in total which are classified into 43 categories with unbalanced distribution between the categories.

## Methods used

1. IDSIA MCDNN Network : [Multi Column DNN](http://people.idsia.ch/~juergen/nn2012traffic.pdf)
2. Contrast Limited Adaptive Histogram Equilization: [CLAHE](https://medium.com/@wolfapple/traffic-sign-recognition-2b0c3835e104)
3. Spatial Transformer Networks: [STN](http://torch.ch/blog/2015/09/07/spatial_transformers.html)
4. Adam and Adabound Optimizer
5. BatchNorm
6. Dropout
7. Data Augmentation
8. Ensemble Methods

## Architectures:

1. Net1 : IDSIA, 3 Conv, 2 FC, Batchnorm and Dropout
2. Net2 : IDSIA, 3 Conv, 2 FC, Batchnorm, 1 STN
3. Net3 : IDSIA, 3 Conv, 2 FC, Batchnorm, Dropout, 1 STN
4. Net4 : IDSIA, 3 Conv, 2 FC, Batchnorm, Dropout, 2 STNs
5. Net5 : IDSIA, 3 Conv, 2 FC, Batchnorm, Dropout, 2 STNs with Batchnorm


## Results:
Ensemble of all 5 => 99.6% test accuracy

## Training:
Use the following command to train-

```python main.py --data data --batch-size batch_size --lr lr --epochs epochs```

Where:
data specifies the name of the data folder
lr is desired learning rate

## Evaluation:
make a folder titled best_models and save your best models' .pth files there, then run
```python evaluate.py```


