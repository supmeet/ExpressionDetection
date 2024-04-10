# Face Expression Recognition Project

This repository contains the code and models for identifying facial expressions using various machine learning and deep learning techniques. The project includes models deployed with Flask/Gunicorn and Streamlit or HuggingFace, along with a dataset for training and testing.

## Models Included

1. **Transfer Learning with Additional Layers**: Model trained on top of pre-trained EfficientNet/VGG with additional layers.
2. **Transfer Learning with Unfreezing Layers**: Model trained on top of pre-trained EfficientNet/VGG by unfreezing some existing layers.
3. **CNN Model**: Convolutional Neural Network model to identify facial expressions directly.
4. **GAN Model**: Generative Adversarial Network model for facial expression generation.
5. **VAE Model**: Variational Autoencoder model for facial expression generation.

## Dataset

The dataset used for training and testing the models can be found in the [FaceExpressions.zip](https://tbcollege0-my.sharepoint.com/personal/bhavik_gandhi_tbcollege_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fbhavik%5Fgandhi%5Ftbcollege%5Fcom%2FDocuments%2FDatasets%2FFaceExpressions%2Ezip&parent=%2Fpersonal%2Fbhavik%5Fgandhi%5Ftbcollege%5Fcom%2FDocuments%2FDatasets&ga=1) file.

## Directory Structure
1. models/: Contains the trained model files and deployment code.
2. datasets/: Holds the dataset used for training and testing.
3. templates/: HTML Files to be used for the Flask App
4. notebooks/: Jupyter notebooks used for training and testing the models.
5. requirements.txt: List of Python dependencies required for the project.
6. README.md: This file providing an overview of the project.

## Video Demo
A video demo showcasing the functionality of the deployed models can be found here [Demo Video](https://azureloyalistcollege-my.sharepoint.com/:v:/g/personal/meetgautambhaipat_loyalistcollege_com/EZ2BXQ5yptJAo4ZA7Jfg0kYBSRQpefMqZcspnMN2JyLrJQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=WuClDx) .

## Contributors
1. Meet Patel
2. Jhanvi Gandhi
3. Priyank Patel
4. Kuldip Mangrola
5. Rutvik Desai
6. Priti Nath
7. Chinmay Parmar
8. Harpreet Kaur Bhatia
9. Simran
