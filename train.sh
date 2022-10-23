#!/bin/bash


python train.py --pretrained_path "../yolov5-6.2/yolov5s.pt" \
--batch_size=16 \
--device="cuda:0" \
--epochs=200 \
--data_info="yamls/data_info.yaml" 


