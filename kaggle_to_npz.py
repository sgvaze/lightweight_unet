from PIL import Image
import numpy as np
import os

# TODO: Insert root path for kaggle images, and output directory for numpy data
IMAGE_DIR = ''
SAVE_PATH = ''

IMAGE_ROWS = 420
IMAGE_COLS = 580

resized_h = 64
resized_w = 128

IMAGES = [i for i in os.listdir(IMAGE_DIR) if 'mask' not in i]
MASKS = [i for i in os.listdir(IMAGE_DIR) if 'mask' in i]

all_ims = np.zeros((len(IMAGES), resized_h, resized_w))
all_masks = np.zeros_like(all_ims)

count = 0

for i, im_file in enumerate(IMAGES):

    mask_file = im_file[:-4] + '_mask' + '.tif'

    im = Image.open(os.path.join(IMAGE_DIR, im_file))
    msk = Image.open(os.path.join(IMAGE_DIR, mask_file))

    im_arr = np.array(im)
    msk_arr = np.array(msk)

    if msk_arr.sum() > 0:

        im = im.resize((resized_w, resized_h))
        msk = msk.resize((resized_w, resized_h))

        im_arr = np.array(im)
        msk_arr = np.array(msk)

        msk_arr[msk_arr >= 0.5] = 1
        msk_arr[msk_arr < 0.5] = 0

        all_ims[count, :, :] = im_arr
        all_masks[count, :, :] = msk_arr

        count += 1

        if count % 500 == 0:
            print("{} out of ~ 2200 images processed".format(count))

all_ims = all_ims[:count, :, :]
all_masks = all_masks[:count, :, :]

all_ims = np.expand_dims(all_ims, axis = 3)
all_masks = np.expand_dims(all_masks, axis = 3)

print(all_ims.shape)
print(all_masks.shape)

np.savez(SAVE_PATH, all_ims, all_masks)