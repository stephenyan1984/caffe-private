#!/bin/bash

IMAGE_DIR=/home/zyan3/local/data/voc12_SBD/img/
LABEL_DIR=/home/zyan3/local/data/voc12_SBD/SegmentationClassText/
TRAIN_FILE_LIST=./examples/voc12_SBD/train_sort_by_aspect_ratio.txt
TEST_FILE_LIST=./examples/voc12_SBD/voc12_SBD/val.txt

# MIN_HEIGHT=378
# MIN_WIDTH=501

# MIN_HEIGHT=375
# MIN_WIDTH=500

MIN_HEIGHT=306
MIN_WIDTH=306

# MIN_HEIGHT=320
# MIN_WIDTH=320

TRAIN_DB_NAME=./examples/voc12_SBD/train_minH_${MIN_HEIGHT}_minW_${MIN_WIDTH}_shuffle_lmdb
TEST_DB_NAME=./examples/voc12_SBD/val_minH_${MIN_HEIGHT}_minW_${MIN_WIDTH}_lmdb


rm -r $TRAIN_DB_NAME
GLOG_logtostderr=1 ./build/examples/voc12_SBD/create_voc12_SBD_dataset.bin \
--min_height=$MIN_HEIGHT --min_width=$MIN_WIDTH --image_dir=$IMAGE_DIR \
--label_dir=$LABEL_DIR --image_list_file=$TRAIN_FILE_LIST --out_database_path=$TRAIN_DB_NAME --shuffle

rm -r $TEST_DB_NAME
GLOG_logtostderr=1 ./build/examples/voc12_SBD/create_voc12_SBD_dataset.bin \
--min_height=$MIN_HEIGHT --min_width=$MIN_WIDTH --image_dir=$IMAGE_DIR \
--label_dir=$LABEL_DIR --image_list_file=$TEST_FILE_LIST --out_database_path=$TEST_DB_NAME --image_name_as_key
