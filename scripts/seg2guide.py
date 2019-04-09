import cv2
import os
import argparse
import numpy as np
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("--input_image", type=str, help="input image")
parser.add_argument("--output_guide_dir", type=str,
                    help="output guides dir path")
parser.add_argument("--output_hdf5_path", type=str, help="output hdf5 path")
args = parser.parse_args()


def make_guides(img_label):
    h, w = img_label.shape[:2]
    guide_hair = np.zeros((h, w))
    guide_face = np.zeros((h, w))
    guide_bg = np.zeros((h, w))
    guide_hair[img_label[:, :, -1] > 50] = 255
    guide_face[img_label[:, :, -2] > 50] = 255
    guide_bg[img_label.sum(axis=-1) < 50] = 255
    return guide_hair, guide_face, guide_bg


def make_hdf5(guides_list: list, hdf5_fpath):
    guides = []
    for guide in guides_list:
        if guide.shape[-1] == 3:
            guide = guide[:, :, 0]
        guides.append(guide)
    guides = np.dstack(guides).transpose(2, 0, 1)

    f = h5py.File(hdf5_fpath, 'w')
    f.create_dataset('guides', data=guides)
    f.close()


if __name__ == "__main__":
    img_label = cv2.imread(args.input_image)
    guide_hair, guide_face, guide_bg = make_guides(img_label)

    if not os.path.exists(args.output_guide_dir):
        os.mkdir(args.output_guide_dir)

    file_name = os.path.basename(args.input_image).split('.')[0]
    guide_hair_file = os.path.join(
        args.output_guide_dir, file_name + '_hair.png')
    guide_face_file = os.path.join(
        args.output_guide_dir, file_name + '_face.png')
    guide_bg_file = os.path.join(args.output_guide_dir, file_name + '_bg.png')

    cv2.imwrite(guide_hair_file, guide_hair)
    cv2.imwrite(guide_face_file, guide_face)
    cv2.imwrite(guide_bg_file, guide_bg)

    make_hdf5([guide_hair, guide_face, guide_bg], args.output_hdf5_path)
    print('Saved guides: ', args.output_guide_dir)
    print('Saved hdf5:', args.output_hdf5_path)
