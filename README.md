# Event-based Object Tracking by Joint Detection
Jessica Lee - Final Project for 15-463

## Abstract
Event cameras are new sensors that record log-changes in brightness for each pixel independently. This implies that all pixels operate asynchronously which give event cameras a significant benefit, including capturing motion at a high frame rate, countering motion blur. Additionally, these cameras have high dynamic range and have low power usage when there is no motion in the scene. Even given these significant advantages, their performance lags behind standard RGB cameras because of their cost, lack of standardized datasets, and unique spatio-temporal data which neural networks are not suited to learn from yet. Additionally, they lack the 30 years of research that frame-based vision is supported by. In this paper, we would like to transfer our RGB frame-based knowledge towards event-data in one of the most basic computer vision tasks, object tracking. We build upon prior work in event-based object detection, and extend it to tracking in a simple manner by adopting the CenterTrack architecture, which is a \textit{tracking-by-detection} framework. First, we utilize Event Volumes in order to create dense input images for the standard neural network. Then, we utilize the CenterTrack architecture and train from scratch.


## What to look for
- src/lib/dataset/datasets/prophese_event.py
This script has all the event code generation
- src/lib/dataset/event_generic_dataset.py
This code handles all the data augmentation, and I changed it to be able to work for streams
- run_training.sh
Runs the training code
