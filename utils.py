import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


SHAPE = (768, 768)


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted: [start0] [length0] [start1] [length1]... in 1d array
    '''
    # reshape to 1d array
    pixels = img.T.flatten()  # Needed to align to RLE direction
    # pads the head & the tail with 0 & converts to ndarray
    pixels = np.concatenate([[0], pixels, [0]])
    # gets all start(0->1) & end(1->0) positions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # transforms end positions to lengths
    runs[1::2] -= runs[::2]
    # converts to the string formatted: '[s0] [l0] [s1] [l1]...'
    return ' '.join(str(x) for x in runs)


# def multi_rle_encode(img):
#     labels = label(img[:, :, 0])
#     return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]


def rle_decode(mask_rle, shape=SHAPE):
    '''
    mask_rle: run-length as string formatted: [start0] [length0] [start1] [length1]... in 1d array
    shape: (height,width) of array to return
    Returns numpy array according to the shape, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    # gets starts & lengths 1d arrays
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    # gets ends 1d array
    ends = starts + lengths
    # creates blank mask image 1d array
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    # sets mark pixels
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # reshape as a 2d mask image
    return img.reshape(shape).T


def masks_as_image(in_mask_list, shape=SHAPE):
    '''Take the individual ship masks and create a single mask array for all ships
    in_mask_list: pd Series: [idx0] [RLE string0]...
    Returns numpy array as (shape.h, shape.w, 1)
    '''
    all_masks = np.zeros(shape, dtype=np.int16)
    # if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def prepare_data():
    masks = pd.read_csv(os.path.join('D:\\rabohii stol\\airbus-ship-detection', 'train_ship_segmentations_v2.csv'))
    # check if a mask has a ship
    masks['ships'] = masks['EncodedPixels'].map(lambda encoded_pixels: 1 if isinstance(encoded_pixels, str) else 0)
    # sum ship# by ImageId and create the unique image id/mask list
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'})
    unique_img_ids['RleMaskList'] = masks.groupby('ImageId')['EncodedPixels'].apply(list)
    unique_img_ids = unique_img_ids.reset_index()

    # Only care image with ships
    unique_img_ids = unique_img_ids[unique_img_ids['ships'] > 0]
    train_ids, val_ids = train_test_split(unique_img_ids,
                                          test_size=.2,
                                          stratify=unique_img_ids['ships'])
    train_ids, val_ids = train_ids.to_numpy(), val_ids.to_numpy()
    return train_ids, val_ids
