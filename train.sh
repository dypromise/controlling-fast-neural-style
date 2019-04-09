#!/bin/bash
# ld libcuda_utils.so
export LD_LIBRARY_PATH="./fast_neural_style/:$LD_LIBRARY_PATH"

time th train.lua \
    -h5_file /media/bigdrive/dingyang/COCO2014/ms-coco-512.h5 \
    -style_image /home/dingyang/style_imgs/Moneph.jpg \
    -style_image_size 512 \
    -style_weights 10 \
    -style_target_type gram \
    -num_iterations 40000 \
    -learning_rate 0.001 \
    -lr_decay_every 4000 \
    -lr_decay_factor 0.8 \
    -batch_size 1 \
    -gpu 0 \
    -display_port 8887 \
    -depth_weights 5 \
#    -style_image_guides ~/pt_test/hd_ori_guide.hdf5
#    -histogram_layers "4,23" \
#    -histogram_weights 1.0 \
