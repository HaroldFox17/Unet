import tensorflow as tf
from losses import dice_p_bce

"""Class implements model architecture"""


class UNet:
    def __init__(self, input_width, input_height, num_channels):
        self.input_width = input_width
        self.input_height = input_height
        self.num_channels = num_channels

    def conv1(self, x):
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        c1 = tf.keras.layers.Dropout(.1)(c1)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        c1 = tf.keras.layers.Dropout(.1)(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        return c1, p1

    def conv2(self, x):
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        c2 = tf.keras.layers.Dropout(.1)(c2)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        c2 = tf.keras.layers.Dropout(.1)(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        return c2, p2

    def conv3(self, x):
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        c3 = tf.keras.layers.Dropout(.1)(c3)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        c3 = tf.keras.layers.Dropout(.1)(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        return c3, p3

    def conv4(self, x):
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        c4 = tf.keras.layers.Dropout(.1)(c4)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        c4 = tf.keras.layers.Dropout(.1)(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
        return c4, p4

    def conv5(self, x):
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        c5 = tf.keras.layers.Dropout(.1)(c5)
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        c5 = tf.keras.layers.Dropout(.1)(c5)
        c5 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2))(c5)
        return c5

    def create_model(self):
        """Creates model according to architecture and compiles it using dice loss and Adam optimizer"""
        inputs = tf.keras.layers.Input((self.input_width, self.input_height, self.num_channels))
        inputs = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
        c1, p1 = self.conv1(inputs)
        c2, p2 = self.conv2(p1)
        c3, p3 = self.conv3(p2)
        c4, p4 = self.conv4(p3)
        c5 = self.conv5(p4)
        u6 = tf.keras.layers.concatenate([c5, c4])
        u6, _ = self.conv4(u6)
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2))(u6)
        u7 = tf.keras.layers.concatenate([u6, c3])
        u7, _ = self.conv3(u7)
        u7 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2))(u7)
        u8 = tf.keras.layers.concatenate([u7, c2])
        u8, _ = self.conv2(u8)
        u8 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2))(u8)
        u9 = tf.keras.layers.concatenate([u8, c1])
        u9, _ = self.conv1(u9)
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u9)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss=dice_p_bce, metrics=['accuracy'])
        return model


# unet = UNet(768, 768, 3)
# model = unet.create_model()
# model.compile(loss=dice_p_bce, optimizer='adam')
# model.summary()
