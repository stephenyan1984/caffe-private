#!/bin/bash

CHECKPOINT_DIR=./examples/voc12_SBD/VGG_16_layer_seg

if [ ! -d "$CHECKPOINT_DIR" ]; then
	mkdir $CHECKPOINT_DIR
fi

GLOG_logtostderr=1 \
# GLOG_minloglevel=1 \
./build/tools/caffe train \
--solver=./examples/voc12_SBD/VGG_16_layer_fc8_fast_LR_batch20_seg_solver.prototxt \
--weights=./models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel