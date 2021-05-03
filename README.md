# FaceNet_EmbeddedSystem
System and Library developed for the inference of an artificial neural network, especially for FaceNet. Made in C and C++ code, it has been tested on an ESP32 microcontroller and raspberry PI.

## Requirements
* `opencv` 4.0 or later
## DeepLearning Functions:

* matrix_NHWC_alloc : Allocate memory for NHWC data type
* cmo_lib_free : Free up memory
* cmo_NHWC_l2_normalize : L2 normalization
* cmo_NHWC_MaxPooling : Maxpooling
* cmo_NHWC_AveragePooling: Average Pooling
* cmo_NHWC_conv :  Convolution between two volumes
* cmo_NHWC_dense : fully conected
* cmo_NHWC_batch_normalize: batch normalization
* cmo_NHWC_concat: Concatenation
* cmo_NHWC_padding: Applies zero margins to volume
* cmo_NHWC_ActivationRelu: Relu activation function


## Instructions
1. Clone the repository.
2. In the src folder, download the weights, the link of the FaceNet weights is in the same folder in the file readme2.md
4. Compile all the files in this folder with some compiler, it was tested with the cross gcc compiler.
5. The program has two options:
   * a. Option == 1: Add a new person, it asks for the name and then takes a photo where you can decide whether or not to save the name for that image, if you choose not, the process will be repeated again.
   * b. option == 2: Identify the face of the person by comparing each encode that is stored in the database.
git clone https://github.com/Itseez/opencv_contrib.git
