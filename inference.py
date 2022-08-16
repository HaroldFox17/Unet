import os
import tensorflow as tf
from utils import rle_encode
from skimage.io import imread
import pandas as pd


test_path = 'D:\\rabohii stol\\airbus-ship-detection\\test_v2'

unet = tf.keras.models.load_model('saved_model/unet')

test_pics = os.listdir(test_path)

inference_df = pd.DataFrame(columns=["ImageId", "EncodedPixels"])

for n, pic in enumerate(test_pics):
    im = imread(os.path.join(test_path, pic))[:, :, :3]
    res = rle_encode(unet(im).numpy())
    inference_df.loc[n] = [pic, res]


inference_df.to_csv("results.csv")
