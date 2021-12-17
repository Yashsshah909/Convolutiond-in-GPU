# Convolutions-in-GPU
Convolution is a very popular technique that is used for signal processing, image processing
and is one of the very first steps in any deep learning algorithm. Convolutional layers are the
major building blocks used in convolutional neural networks.
Mathematically convolution is the treatment of a matrix by a smaller matrix known as the ‘Kernel’
which decides the effect you want on the image. Effects on images like blurring, edge detection,
sharpening, etc all use convolution of images. An image is a matrix of pixels and when a much
smaller matrix known as the ‘Kernel’ is overlaid on the image then we get modified images. It is
determined by the value of the central pixel by adding the weighted values of the center element
and all its neighbouring elements. The type of modification depends on the kernel used.
As convolution contains a lot of arithmetic operations (multiplication and additions), performing
such operations on the GPU devices, that are specifically designed for image and graphics
processing that would increase the performance of the overall system.


In this project we will implement the convolution algorithm a total of 3 times, in the CPU, in the
GPU with the use of constant memory and in the GPU with the use of shared memory.
