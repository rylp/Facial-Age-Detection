
# Age Classification using CNN

## Overview
A Convolutional Neural Network designed from scratch trained using Keras framework that categorizes images of people based on their ages. 
The model categorizes the input image based on three categories- Young , Middle and Old. 

## Dataset
**IMFDB** - Indian Movie Face Database was the dataset used for this project.
IMFDB is a large unconstrained face database consisting of 34512 images of 100 Indian actors collected from more than 100 videos. All the images are manually selected and cropped from the video frames resulting in a high degree of variability interms of scale, pose, expression, illumination, age, resolution, occlusion, and makeup.
url- http://cvit.iiit.ac.in/projects/IMFDB/

## Preprocessing
The following preprocessing was applied to each image:

- Have trained the network on frontal faces images
- Random crops of 64 × 64 pixels from the input image of random sizes
- Randomly mirror images in each forward-backward training pass
- Data Augmentation is used

## Accuracy-Loss Trade-off Graphs

![With 150 epochs](accuracy_loss_tradeoff_graphs/150epochs-v15.png)
![With 100 epochs](accuracy_loss_tradeoff_graphs/100_epochs.png)

## Libraries Used
1. OpenCV</br>
2. Keras</br>
3. Numpy</br>
4. Pandas</br>
5. Seaborn</br>
6. Matplotlib</br>
7. Pickle</br>
8. sklearn</br>
9. imutils</br>

## Results

Training Accuracy : **93.30%**</br>
Validation Accuracy : **91.26%**</br>

## Outputs

![](output_images/sample_young_image.png)
![](output_images/sample_middle_image.png)
![](output_images/sample_old_image_now.png)
![](output_images/sample_young_image3.png)
![](output_images/sample_young_image2.png)
![](output_images/sample_old_image2.png)

## Contributors
-Rohan Limaye: https://github.com/rylp </br>
-Rohan Naik: https://github.com/rohan-naik07 

