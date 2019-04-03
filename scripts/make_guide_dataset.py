import os
# import json
import argparse
from threading import Thread
from Queue import Queue

import numpy as np
from scipy.misc import imread, imresize
import h5py
from skimage.color import rgb2grey
from skimage import img_as_ubyte

"""
Create an HDF5 file of images for training a feedforward style transfer model.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='data/coco/images/train2014')
parser.add_argument('--train_guides_dir', default='data/coco/images/train2014')
parser.add_argument('--val_dir', default='data/coco/images/val2014')
parser.add_argument('--val_guides_dir', default='data/coco/images/val2014')
parser.add_argument('--output_file', default='data/ms-coco-512.h5')

parser.add_argument('--num_guides', type=int, default=3)
parser.add_argument('--height', type=int, default=512)
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--max_images', type=int, default=-1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--include_val', type=int, default=1)
parser.add_argument('--max_resize', default=16, type=int)
parser.add_argument('--lum', type=bool, default=False)
args = parser.parse_args()


def add_data(h5_file, image_dir, image_guide_dir, prefix, args):
    # Make a list of all images in the source directory
    image_list = []
    image_guide_list = []
    guide_str = ['hair', 'face', 'bg']
    image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    for filename in os.listdir(image_dir):
        ext = os.path.splitext(filename)[1]
        if ext in image_extensions:
            this_image_guides = [
                os.path.join(image_guide_dir, os.path.splitext(
                    filename)[0] + '_' + x + '.png')
                for x in guide_str]
            if not os.path.exists(this_image_guides[0]):
                continue
            image_guide_list.append(this_image_guides)
            image_list.append(os.path.join(image_dir, filename))

    if args.max_images > 0:
        image_list = image_list[:args.max_images]
    num_images = len(image_list)

    # Create dset for images
    dset_name = os.path.join(prefix, 'images')
    dset_size = (num_images, 3, args.height, args.width)
    imgs_dset = h5_file.create_dataset(dset_name, dset_size, np.uint8)

    # Create dset for guides
    guide_set_name = os.path.join(prefix, 'guides')
    guide_set_size = (num_images, args.num_guides, args.height, args.width)
    guides_dset = h5_file.create_dataset(
        guide_set_name, guide_set_size, np.uint8)

    # input_queue stores (idx, filename) tuples,
    # output_queue stores (idx, resized_img) tuples
    input_queue = Queue()
    output_queue = Queue()

    # Read workers pull images off disk and resize them
    def read_worker():

        def _crop(img):
            H, W = img.shape[0], img.shape[1]
            H_crop = H - H % args.max_resize
            W_crop = W - W % args.max_resize
            img = img[:H_crop, :W_crop]
            img = imresize(img, (args.height, args.width))
            return img

        while True:
            idx, filename, guide_filenames = input_queue.get()
            img = imread(filename)
            guides = []
            for f in guide_filenames:
                g = imread(f)
                if g.ndim == 3:
                    guides.append(g[:, :, 0])
                elif g.ndim == 2:
                    guides.append(g)

            # First crop the image so its size is a multiple of max_resize
            try:
                img = _crop(img)
                guides = [_crop(g) for g in guides]
            except (ValueError, IndexError) as e:
                print(filename)
                print(img.shape, img.dtype)
                print(e)
            input_queue.task_done()
            output_queue.put((idx, img, guides))

    # Write workers write resized images to the hdf5 file
    def write_worker():
        num_written = 0
        while True:
            idx, img, guides = output_queue.get()
            if img.ndim == 3:
                # RGB image, transpose from H x W x C to C x H x W
                imgs_dset[idx] = img.transpose(2, 0, 1)
            elif img.ndim == 2:
                # Grayscale image;
                imgs_dset[idx] = img
            guide = np.asarray(guides, dtype='int')
            guides_dset[idx] = guide
            if args.lum:
                # Make greyscale dataset
                img = img_as_ubyte(
                    np.tile(rgb2grey(img)[None, :, :], (3, 1, 1)))
                imgs_dset[idx] = img
            output_queue.task_done()
            num_written = num_written + 1
            if num_written % 100 == 0:
                print('Copied %d / %d images' % (num_written, num_images))

    # Start the read workers.
    for i in range(args.num_workers):
        t = Thread(target=read_worker)
        t.daemon = True
        t.start()

    # h5py locks internally, so we can only use a single write worker =(
    t = Thread(target=write_worker)
    t.daemon = True
    t.start()

    for idx, filename in enumerate(image_list):
        if args.max_images > 0 and idx >= args.max_images:
            break
        guides_filenames = image_guide_list[idx]
        input_queue.put((idx, filename, guides_filenames))

    input_queue.join()
    output_queue.join()


if __name__ == '__main__':

    with h5py.File(args.output_file, 'w') as f:
        add_data(f, args.train_dir, args.train_guides_dir, 'train2014', args)

        if args.include_val != 0:
            add_data(f, args.val_dir, args.val_guides_dir, 'val2014', args)
