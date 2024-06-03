#!/bin/bash

# Download pretrained detection model
wget -P pretrained_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar
# Download pretrained recognition model
wget -P pretrained_models/ https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar

# Extract the downloaded files and clean up
cd ./pretrained_models
tar -xf en_PP-OCRv3_det_distill_train.tar && rm -rf en_PP-OCRv3_det_distill_train.tar
tar -xf en_PP-OCRv4_rec_train.tar && rm -rf en_PP-OCRv4_rec_train.tar
cd ..


