#!/bin/bash
time th train.lua \
    -h5_file /media/bigdrive/dingyang/COCO2014/ms-coco-512.h5 \
    -style_image /home/dingyang/style_imgs/Moneph.jpg \
    -style_image_size 600 \
    -style_weights 10 \
    -style_target_type gram \
    -style_image_guides /home/dingyang/pt_test/hd2_guides.hdf5 \
    -num_iterations 80000 \
    -batch_size 1 \
    -gpu 0
