# Brain Tumor Detection

# Overview
Diagnosing brain scans quickly and accurately is a critical service to society. Because it can take medical experts an extended period of time to diagnose a set of scans, we trained a convolutional neural network to diagnose brain scans as either beneign, malignant, or medically normal. We trained and applied this network to a novel dataset from Kaggle.

# Model
Our model is comprised of four convolutional blocks, each with two Conv2D layers followed by max pooling. The number of filters for each convolutional layer increases by a factor of 2 with every block. We employ batch normalization after each block to ensure model stability and dropout layers after the third and fourth convolutional blocks to prevent overfitting. Adam is used for learning rate optimization. Additionally, we employ a dynamic learning rate: we decrease the learning rate by a factor of 2 if log of validation loss does not improve after two consecutive epochs.

# Results


# Contributors
@hamzah-shah - Hamzah Shah
@bach-z - Bashar Zaidat
