#!/bin/bash

CHECKPOINT_DIR=./examples/stanford_background/renet_3_lay

if [ ! -d "$CHECKPOINT_DIR" ]; then
	mkdir $CHECKPOINT_DIR
fi

GLOG_logtostderr=1 \
# GLOG_minloglevel=1 \
./build/tools/caffe train \
--solver=./examples/stanford_background/renet_3_lay_HP_nolbw_nopeep_solver.prototxt