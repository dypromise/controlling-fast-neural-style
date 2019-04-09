source activate py27
python make_guide_dataset.py \
    --train_dir /media/bigdrive/dingyang/our_faces \
    --train_guides_dir /media/bigdrive/dingyang/our_faces_guides \
    --val_dir /media/bigdrive/dingyang/our_faces_val/ \
    --val_guides_dir /media/bigdrive/dingyang/our_faces_val_guides/ \
    --output_file /media/bigdrive/dingyang/ourfaces_512.hdf5 \
    --height 512 \
    --width 512 \
    --num_guides 3
