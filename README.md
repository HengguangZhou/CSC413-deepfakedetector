# CSC413-deepfakedetector

### Description
This is a course project for CSC413/2516 Winter 2021 Neural Networks and Deep Learning.
We implemented a method that can determine if an image is real or artificial by comparing its distance to a reference real image in high-dimensional feature space. By learning common real features in real images, our method can generalize to distinguish unseen images

### Dataset
You can get the data from https://www.kaggle.com/xhlulu/140k-real-and-fake-faces/download

### Training
To train the model with default hyperparameters
```
python train.py --train_data /real_vs_fake/real-vs-fake/train --val_data /real_vs_fake/real-vs-fake/valid
```

### Testing
After training, you can evaluate the performance on single image pair by the following code:
```
python test.py --real_image /path/to/reference_image/ --unknown_image /path/to/test_image/
```
