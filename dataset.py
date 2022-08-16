from utils import rle_decode, prepare_data
import tensorflow as tf
import os
from skimage.io import imread
from skimage.transform import resize

"""Creating datasets from generators to train model on"""

train_path = 'D:\\rabohii stol\\airbus-ship-detection\\train_v2'  # Used absolute path must be changed to reimplement
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 768, 768, 3

train_ids, test_ids = prepare_data()


def gen_pairs_train():
    for i in train_ids:
        pic = imread(os.path.join(train_path, i[0]))[:, :, :IMG_CHANNELS]
        pic = resize(pic, (IMG_HEIGHT, IMG_WIDTH))
        # pic = tf.Tensor(pic)
        mask = rle_decode(i[2][0])
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        yield pic, mask


def gen_pairs_test():
    for i in test_ids:
        pic = imread(os.path.join(train_path, i[0]))[:, :, :IMG_CHANNELS]
        pic = resize(pic, (IMG_HEIGHT, IMG_WIDTH))
        # pic = tf.Tensor(pic)
        mask = rle_decode(i[2][0])
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        yield pic, mask


train_dataset = tf.data.Dataset.from_generator(gen_pairs_train, output_types=(tf.float16, tf.float16),
                                               output_shapes=([768, 768, 3], [768, 768]))
train_dataset = train_dataset.batch(16)
val_dataset = tf.data.Dataset.from_generator(gen_pairs_test, output_types=(tf.float16, tf.float16),
                                             output_shapes=([768, 768, 3], [768, 768]))
val_dataset = val_dataset.batch(16)

# p, m = next(iter(train_dataset))
# print(p.shape, m.shape)
