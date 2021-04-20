# FaceNet_EmbeddedSystem
System and Library developed for the inference of an artificial neural network, especially for FaceNet. Made in C code, it has been tested on an ESP32 microcontroller and raspberry PI.

## Architecture Facenet 
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
