# Machine Learning Projects
This repository contains all of my machine learning projects from Summer 2021. A brief description of each project can be found below. I tried to explain what I as doing in detail throughout each project.

# ASL Classifier
An ASL Classifier neural network that can detect signs from the American Sign Language alphabet, including space, del, and nothing. It uses a standard neural network architecture and achieved 97% accuracy in mmy tests. Potential future work includes real time integration with Snapchat via Lens Studio

# DCGAN.ipynb
Deep Convolutional Generative Adverserial Network (DGCAN) designed to learn to replicate landscape paintings from [this](https://www.kaggle.com/ipythonx/wikiart-gangogh-creating-art-gan) dataset. 

Architecture based on [this](https://www.kaggle.com/jadeblue/dcgans-and-techniques-to-optimize-them) tutorial, but many modifications were made along the way. 

Trained for 6 hours, totalling 36 epochs in Google Colab. Runtime kept disconnecting, so I could not get super great results. I think it could do better with more training, but I want to try reworking it as a Progressive Growth GAN (PGGAN), found in another Colab file. 

# DCGAN (folder)
The same model as the Colab notebook, but I converted the code to traditional python stuff. I ran it a couple times for 2.5 hours, totalling 300 epochs and achieved decent results actually.

Final results from the best round of testing
![Final](https://user-images.githubusercontent.com/58013394/125630841-6069329a-eae5-4945-a8c4-0de045228a0f.png)

More images can be found in various version folders within DCGAN.

# PGGAN.ipynb
Progressive Growth Generative Adverserial Network (PGCAN) designed to learn to replicate landscape paintings from [this](https://www.kaggle.com/ipythonx/wikiart-gangogh-creating-art-gan) dataset. 

Architecture based on [this](https://fabulousjeong.medium.com/tensorflow2-0-pggan-progressive-growing-of-gans-for-improved-quality-stability-and-variation-67a474b39356) tutorial. Still a work in progress. Once I get it working, I'm hoping to make some changes to it to make it mine

# PGGAN (folder)
Standard python version of PGGAN.ipynb. Improved architecture, statistic collection, etc. (WIP)

# CIFAR10_classifier.ipynb
Simple convolutional network built in Pytorch to identify and classify images from the CIFAR10 dataset

