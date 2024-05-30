#!/bin/bash

# Step 1: Download the zip file
wget https://guillaumejaume.github.io/FUNSD/dataset.zip

# Step 2: Unzip the downloaded file
unzip dataset.zip
rm -rf __MACOSX

# Step 3: Rename the directories
mv dataset/training_data dataset/train
mv dataset/testing_data dataset/test

echo "Dataset download and setup complete."
