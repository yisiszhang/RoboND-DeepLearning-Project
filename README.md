## Project: Follow me


---


[//]: # (Image References)

[image1]: ./architecture.png

## Overview
This project uses a fully convolutional network (FCN) for semantic segmentation of images taken from a simulated drone camera. The goal is to identify a target person from different angles and distances, non-target people and the background, and to have the drone follow the target.

The model was trained on Amazon AWS and a final score of 0.52 was achieved in the end. The details of the model is explains as follows.

## Contents
####[-- Architecture](#architecture)
####[-- Hyperparameters](#hyperparameters)
####[-- Performance](#performance)
####[-- Future Enhancements](#future-enhancement)

---
### Architecture

FCN is a powerful tool for semantic segmentation as it extracts high-dimensional features from the image as well as preserves the spatial information. Thus, individual image pixels can be categorized based on its feature and the output images share the same dimensions in width and heights as the input images. The architecture of the network is shown in Figure 1. Overall, the architecture is comprised of a three-layer encoder network followed by a three-layer decoder network. A 1x1 convolution layer is added before and after the encoder to increase leaning depth. Finally a convolution layer with stride of 1 and softmax activation is attached to produce the segmented image output. The filter size increases through the encoder from 32 to 128, while the w/h dimensions are halved every time the encoder is performed. The decoder steps reverse this process.

![alt text][image1]
Figure 1

#### 1. Encoder
The encoder section is a classification network which gradually builds more "insightful" representation of the input image through feed-forward layers. Using a kernel size of 3 and stride of 2 yields a half-sized width/height for the next layer. The filter sizes are set up in ascending order. In this project, three encoders are used. Each encoder block consists of a separable convolution layer and a batch normalization step.

Separable convolution is a more efficient algorithm than regular convolution. For example, for an input of I channels, an output of O channels and a kernel of kxk size, the resulted number of parameters is IxOxkxk (ignoring biases). When separable convolutions are adopted, each of the I input channel is traversed with a kxk kernel, giving I feature maps. Then each feature map is traversed with O 1x1 convolutions and the vectors are added together. This would yield (Ixkxk+OxIx1x1) parameters, significantly smaller than before. A batch normalization layer normalizes the batch to zero mean and unit variance. By doing so, internal variate shift is reduced. This step can accelerate gradient descent.

#### 2. 1x1 Convolution layer
An 1x1 convolution layer is added before and after the encoder steps. This layer has a kernel size and stride of 1 and hence performs the convolution only through the depth, retaining the "pixel-wise" spatial information. The purpose of this layer here is to add a depth of the network. When used in other scenarios, sometimes it reduces dimension through the filter size. It is also a way to increase network depth as each layer is accompanied with a nonlinear activation like ReLU. The 1x1 Convolution layers here in principle are not necessary but improve performance of the model. The output layers of the 1x1 convolution layers are also batch normalized and fed into the next layer.

A 1x1 convolution is also added in the last step of the network to output the segmented image with the depth size the same as the number of classes. Softmax activation is applied to calculate the probability for a pixel to belong a class.

#### 3. Decoder
The decoder gradually recovers the spatial information by taking earlier layers into convolution. Three decoders are used to match the number of encoders. The current layer and a earlier layer with doubled width and height are grouped through "skip connection". The later layer is upsampled in width and height by a factor of two, which matches the size of the connected layer. These two layers are concatenated and then the volume is carried through two layers of separable convolutions with batch normalization. The convolutions have stride of 1 to retain the size of the w/h dimension. The last decoder outputs a volume the same pixel size as the input image.

#### 4. Fully connected layer
While this section title is misleading, there is actually no strictly fully connected layers in a convolutional neural network. There are only 1x1 convolution layers with fully connected table. For example, for a wxhxd input and m output layer, a fully connected layer indicates a kernel size of (wxhxd) with m filters. A 1x1 convolution layer convolutes with wxh kernels of filter size m. Thus a fully connected layer is different from a 1x1 convolution layer.


### Hyperparameters

There are three hyperparameters that are tuned to optimize the model -- learning rate, batch size and number of epochs.

#### 1. Learning rate
Learning rate determines how fast the weights are updated along the negative gradient direction. Larger learning rates usually train a network faster, however it might also fail to converge. Small learning rates do not have the convergence problem but may take too long to reach the global minimum. It may also trap in a local minimum for a long time. However, given enough long time a small learning rate would yield better results than the large learning rate. When tuning this parameter, initially a small learning rate of 0.001 was used, but the model quickly fell into overfitting within 3 epochs. Reflected in the test, a great more false positives in the no-target scenario and false negatives in the target-far scenario were observed. A learning rate of 0.01 reduced the overfitting problem, but more epochs were needed as the validation loss curve converged slower.

#### 2. Batch size
Using batches reduces the computational cost. Larger batch size yields better results, but also takes longer to compute and sometimes may reach the limit of the computing system. I found a batch size of 32 a reasonable choice. A batch size of 16 did not perform well and a batch size of 64 did not make significant improvement on the accuracy but doubled the training time.

#### 3. Number of epochs
More number of epochs will always improve the accuracy. The loss decreases with epoch iterations, but starts leveling off after certain point. I found that the volatility of the validation loss started reducing after 10 epochs and the loss continued dropping slowly. I cut it off at 20 epochs as the loss was relatively low and stable.

### Performance

The performance of the model yields 0.52 final grade score. Overall, it performed quite well for the images taken following behind the target, relatively good for the images without the target, but failed to detect more than 1/3 of the target in the far-away scenario.

### Future Enhancements
The model failed mostly when the target is far away. One reason is that the small object is only a few pixels in a image and the signal to noise is naturally worse. The other reason is that the training size is not comparable to other categories as there are way more samples with the background and non-target people. Adding more training data with only the far away target may improve the performance. In addition, inserting more layers in the encoder section may learn better to extract the feature of the far target. Besides those factors, more epochs may also increase the accuracy.

This data would probably not work well for following the objects that never show up in the training data. To use it for detecting novel objects, a set of train/validation data containing those objects are needed. The model should allow more number of classes. However, since the existing model already extracted basic features, the new training may only need to train part of the layers closer to the output.
