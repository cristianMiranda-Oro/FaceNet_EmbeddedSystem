# FaceNet_EmbeddedSystem
System and Library developed for the inference of an artificial neural network, especially for FaceNet. Made in C and C++ code, it has been tested on an ESP32 microcontroller and raspberry PI.

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

## Facenet (Architecture) 
| Type | Output | Params |
| --- | --- | --- |
| Input | 96x96x3 | 0 |
| Zero_Padding2d | 102x102x3| 0 |
| Conv2D | 48x48x64 | 9472 |
| BatchNormalization |48x48x64 | 256 |
| Activation | 48x48x64| 0 |
|  Zero_Padding2d | 50x50x64 | 0 |
| max_pooling2d |24x24x64| 0 |
| Conv2D | 24x24x64 | 4160 |
| BatchNormalization | 24x24x64 | 256 |
| Activation | 24x24x64 | 0 |
|  Zero_Padding2d | 26x26x64 | 0 |
| Conv2D | 24x24x192 | 110784 |
| BatchNormalization |24x24x192| 768 |
| Activation |24x24x192| 0 |
|  Zero_Padding2d | 26x26x192 | 0 |
| max_pooling2d |12x12x192| 0 |
| Inception_3a | 12x12x256 | 165168 |
| Inception_3b | 12x12x320 | 229568 |
| Inception_3c | 6x6x640 | 399712 |
| Inception_4a | 6x6x640 | 548608 |
| Inception_4e | 3x3x1024 | 719840 |
| Inception_5a | 3x3x96 | 794688 |
| Inception_5b | 3x3x726 | 665664 |
| average_pooling2d | 1x1x736 | 0 |
| flatten | 736 | 0 |
| dense_layer | 128 | 94336 |
| L2 Normalization | 128 | 0 |

## Instructions
1. Clone the repository.
2. In the src folder, download the weights, the link of the FaceNet weights is in the same folder in the file readme2.md
3. This version depends on OpenCV for geolocation of the face in the image delivered by the camera, therefore having Opencv 4V installed
4. Compile all the files in this folder with some compiler, it was tested with the cross gcc compiler.
5. The program has two options:
   * a. Option == 1: Add a new person, it asks for the name and then takes a photo where you can decide whether or not to save the name for that image, if you choose not, the process will be repeated again.
   * b. option == 2: Identify the face of the person by comparing each encode that is stored in the database.
