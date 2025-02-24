Facial Emotion Detection With Transfer Learning 

- Anand Panajkar
- COEP Technological University 

Project Overview:
-----------------
This project aims to detect human emotions from images using deep convolutional neural networks (DCNNs). The models utilized include VGG19 and ResNet50 architectures pretrained on ImageNet. Emotions detected include anger, fear, happiness, sadness, surprise, and neutral.

Methodology
-----------
Dataset
The FER2013 dataset used consists of facial images labeled with one of the six emotions mentioned above. Images are stored as pixel arrays, which are resized to (224, 224) for model input.

Preprocessing
Removed the Disgust Emotion from the dataset as it has the lowest number of images leading to skewed or biased model. All emotions are filtered have equal number of images i.e 4000 randomly sampled from the dataset.
The images are read as 48x48 pixels, resized into 224x224 shape, converted into 3 channels and normalized for compatiblity with the pretrained models. 

Model Training
Two main architectures were employed:

VGG19: Transfer learning approach with pre-trained weights on ImageNet.
ResNet50: Similar transfer learning approach but with a different base architecture.

Training parameters include batch size (8-16), number of epochs (10-15), and early stopping to prevent overfitting. Models are evaluated on validation accuracy and loss, with performance metrics visualized using Matplotlib and Seaborn.

Training Results
Both models achieved similar accuracies on the validation set:
VGG19: 50% validation accuracy after 12 epochs.
ResNet50: 52% validation accuracy after 15 epochs.

Prediction
For prediction, the models are used to detect emotions from new images. Faces are detected using the Haar Cascade classifier, cropped, and then fed into the trained models. Predictions provide the emotion label along with confidence scores.

Constraints
Memory Usage: Due to the large size of VGG19 and ResNet50 architectures, GPU memory requirements can be substantial. Batch size was kept either 8 or 16 to avoid exceeding the memory limit. 
Training the models take upto 20 GB of RAM and 15 GB of GPU P100 (max available).


Files and Directories:
----------------------
1. Codes: 
   - model.py / model.ipynb: Python script to preprocess the dataset, train the models and get predictions.
   - emotion_detection.py: Python script to test run the model. 

2. Weights: 
   - vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5: Pretrained weights for VGG19 model.
   - resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5: Pretrained weights for ResNet50 model.
   - vgg19.h5: Weights of the trained custom vgg19 model 
   - resnet_model.h5: Weights of the trained custom resnet model 

3. Dataset:
   - dataset_description.txt: Description of the dataset used for training/testing.
   - fer2013: Fer2013 standard dataset 
   - testing_data: Dir of raw images for testing the model 

Usage:
------
1. Install dependencies listed in requirements.txt using pip

2. Test the models:
- Use sample images from testing_data or provide your own images for emotion detection.
- Modify the images and model weights path to the correct working directory 
- Run emotion_detection.py to detect faces in images and predict emotions using the trained models.
	

Note:
-----
- Adjust paths and configurations as per your environment setup.


