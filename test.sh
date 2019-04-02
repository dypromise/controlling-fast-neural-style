#!/bin/bash
time th  fast_neural_style.lua \
    -model ./checkpoint.t7 \
    -input_image ~/input_female_crop_models/qianyiwang_crop.png \
    -output_image stylozed.png
