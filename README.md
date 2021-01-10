# Diabetic-Retinopathy-detection
A CNN ensembling based approach to grade severity of DR from fundus photographs

We're a group of B.Tech final year students from NIT Warangal, India. Our final year project aims to automate the process of detecting the stage of DR severity from fundus photographs.

We've developed an ensemble of pre trained models - InceptionNetV3 and DenseNet121. We have also proposed features extraction techniques using digital image processing.

# Procedure
- Converted the problem into a multilabel classification problem and trained our model to maximize the quadratic kappa score. The intuition for treating the problem as a multilabel classification problem comes from the observation that the features present in an image would contain features of lower severity disease images as well.
- Used a weighted ensemble of the 2 models
- Converted the output tensor into a binary tensor indicating the possiblity of belonging to each class -> used thresholding
- To predict class - y.astype(int).sum(axis=1) - 1 

1. Let's say that the numpy array obtained from the last layer of both the models is -
 a(DenseNet121) = [0.97,0.6,0.75,0.4,0.2] and b(InceptionV3) = [0.8,0.77,0.69,0.72,0.35]. Each value in the array depicts the probability of the image belonging to the class corresponding to its index.
2. Weighted average of these 2 arrays [We obtained the best kappa score with weights 0.8 for DenseNet121, 0.2 for InceptionV3] results in - 
 0.8*a + 0.2*b = [0.936,0.634,0.738,0.464,0.23]
3. This is followed by thresholding (Similary, we obtained the best results using 0.5 as threshold) - [1,1,1,0,0]
4. To predict the class from the obtained binary array : 1+1+1+0+0 - 1 = 2 (classes are indexed from 0, hence subtract 1).
 Hence, the image would be classified as class 2.

We were able to achieve a kappa score on 0.897 on the private test set using only images of size 224x224.

# Dataset

Kaggle APTOS 2019 https://www.kaggle.com/c/aptos2019-blindness-detection/data

# Files
1) Ensemble.ipynb - A Keras file with code for training and testing the performance of ensemble model.

2) Augmentation.ipynb - Contains preprocessing and augmentation code for fundus images

3) Blood vessels morpho.ipynb - Contains code for traditional feature extraction for blood vessels

4) micro morpho.ipynb - Contains code for traditional feature extraction for microaneurysms

5) exudates.ipynb - Contains code for traditional feature extraction for exudates
