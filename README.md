<p align="center">
  <img src="https://github.com/bash-z/brain_tumor_detection/blob/main/benign_interpretation_example.jpg?raw=true"/>
  <figcaption>Left. Benign brain tumor Right. Model interpretation of the benign scan: what the network considers most as it updates weights </figcaption>
</p>

# Brain Tumor Detection

# Overview
We report 94% accuracy for a convolutional neural network that diagnoses brain scans as either beneign, malignant, or medically normal. We trained this network using Keras and applied it to a novel dataset from Kaggle. You can view the dataset [here](https://www.kaggle.com/alifrahman/modiified).

# Model
Our model is comprised of four convolutional blocks, each with two Conv2D layers followed by max pooling. The number of filters for each convolutional layer increases by a factor of 2 with every block. We employ batch normalization after each block to ensure model stability and dropout layers after the third and fourth convolutional blocks to prevent overfitting. Adam is used for learning rate optimization. Additionally, we employ a dynamic learning rate: we decrease the learning rate by a factor of 2 if log of validation loss does not improve after two consecutive epochs.

# Model Interpretation
After each epoch, we calculate the gradient of the loss with respect to the input scan and then plot the multiplication of this gradient by the input scan. The plot highlights which areas of the brain scan the network considers most important as it updates the weights before the next epoch.

# Contributors
[@hamzah-shah ](https://github.com/hamzah-shah)- Hamzah Shah  
[@bash-z](https://github.com/bash-z) - Bashar Zaidat
