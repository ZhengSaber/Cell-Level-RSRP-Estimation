import os
import tensorflow as tf
import numpy as np
from tensorflow import keras


# class Downsample(keras.Model):
#
#     def __init__(self, filters, size, strides, pool):
#         super(Downsample, self).__init__()
#
#         initializer = tf.random_normal_initializer(0., 0.02)
#         # self.conv0 = keras.layers.Conv2D(filters,
#         #                                  (size, size),
#         #                                  strides=1,
#         #                                  padding='same',
#         #                                  kernel_initializer=initializer,
#         #                                  use_bias=False)
#         self.conv1 = keras.layers.Conv2D(filters,
#                                          (size, size),
#                                          strides=strides,
#                                          padding='same',
#                                          kernel_initializer=initializer,
#                                          use_bias=False)
#
#         self.MaxPool =  keras.layers.MaxPool2D((pool,pool),strides=pool)
#             # tf.nn.max_pool2d(x, [1, pool, pool, 1], [1, pool, pool, 1], padding='VALID')
#
#     def call(self, x, training):
#         # x = self.conv0(x)
#         x = self.conv1(x)
#         x = tf.nn.relu(x)
#         x = self.MaxPool(x)
#         return x
#
#
# class Upsample(keras.Model):
#
#     def __init__(self, filters, size, apply_dropout=True):
#         super(Upsample, self).__init__()
#
#         initializer = tf.random_normal_initializer(0., 0.02)
#         # self.up_conv_0 = keras.layers.Conv2DTranspose(filters,
#         #                                             (size, size),
#         #                                             strides=1,
#         #                                             padding='same',
#         #                                             kernel_initializer=initializer,
#         #                                             use_bias=False)
#         self.up_conv = keras.layers.Conv2DTranspose(filters,
#                                                     (size, size),
#                                                     strides=2,
#                                                     padding='same',
#                                                     kernel_initializer=initializer,
#                                                     use_bias=False)
#
#
#     def call(self, x1, x2, training=None):
#         # x = self.up_conv_0(x1)
#         x = self.up_conv(x1)
#         x = tf.nn.relu(x)
#         x = tf.concat([x, x2], axis=-1)
#         return x


# class Generator(keras.Model):
#
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         initializer = tf.random_normal_initializer(0., 0.02)
#
#         self.down0 = Downsample(6, 3, 1, 1)
#         self.down00 = Downsample(40, 5, 1, 1)
#
#         self.down1 = Downsample(50, 5, 1, 2)
#         self.down10 = Downsample(60, 5, 1, 1)
#         self.down2 = Downsample(100, 5, 1, 2)
#         self.down20 = Downsample(100, 3, 1, 1)
#         self.down3 = Downsample(150, 5, 1, 2)
#         self.down4 = Downsample(300, 5, 1, 2)
#         self.down5 = Downsample(500, 5, 1, 2)
#         self.down6 = Downsample(500, 5, 1, 2)
#
#
#         self.up1 = Upsample(500, 4)
#         self.up2 = Upsample(300, 4)
#         self.up3 = Upsample(150, 4)
#         self.up30 = Downsample(150, 3, 1, 1)
#         self.up4 = Upsample(100, 4)
#         self.up40 = Downsample(100, 5, 1, 1)
#         self.up5 = Upsample(50, 4)
#
#         self.lastconvT = keras.layers.Conv2DTranspose(20, (4, 4),
#                                                  strides=2,
#                                                  padding='same',
#                                                  kernel_initializer=initializer)
#         self.lastconv = keras.layers.Conv2D(1, (5,5),
#                                                  strides=1,
#                                                  padding='same',
#                                                  kernel_initializer=initializer)
#
#     def call(self, x, training=None):
#         # x shape == (bs, 64, 64, 2)
#         x0 = self.down0(x, training=training)
#         x0 = self.down00(x0, training=training)
#
#         x1 = self.down1(x0, training=training)  # (bs, 32, 32, 16)  (bs, 16, 16, 16)
#         x1 = self.down10(x1, training=training)
#         x2 = self.down2(x1, training=training)  # (bs, 16, 16, 32)    (bs, 8, 8, 16)
#         x2 = self.down20(x2, training=training)
#         x3 = self.down3(x2, training=training)  # (bs, 8, 8, 64)  (bs, 4, 4, 16)
#         x4 = self.down4(x3, training=training)  # (bs, 4, 4, 128)  (bs, 2, 2, 16)
#         x5 = self.down5(x4, training=training)  # (bs, 2, 2, 128)  (bs, 1, 1, 16)
#         x6 = self.down6(x5, training=training)  # (bs, 1, 1, 128)
#
#
#
#         u1 = self.up1(x6, x5, training=training)  # (bs, 2, 2, 128)
#         u2 = self.up2(u1, x4, training=training)  # (bs, 4, 4, 128)
#         u3 = self.up3(u2, x3, training=training)  # (bs, 8, 8, 64)
#         u3 = self.up30(u3, training=training)
#         u4 = self.up4(u3, x2, training=training)  # (bs, 16, 16, 32)
#         u4 = self.up40(u4, training=training)
#         u5 = self.up5(u4, x1, training=training)  # (bs, 32, 32, 16)
#
#         # u1 = self.up1(x5, x4, training=training)  # (bs, 2, 2, 128)
#         # u2 = self.up2(u1, x3, training=training)  # (bs, 4, 4, 128)
#         # u3 = self.up3(u2, x2, training=training)  # (bs, 8, 8, 64)
#         # u4 = self.up4(u3, x1, training=training)  # (bs, 16, 16, 32)
#
#         out = self.lastconvT(u5)  # (bs, 64, 64, 1)
#         out = self.lastconv(out)
#         # out = self.last(u4)  # (bs, 64, 64, 1)
#         # out = tf.nn.tanh(out)
#
#         return out

class Downsample(keras.Model):

    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()

        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = keras.layers.Conv2D(filters,
                                         (size, size),
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(keras.Model):

    def __init__(self, filters, size, apply_dropout=True):
        super(Upsample, self).__init__()

        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = keras.layers.Conv2DTranspose(filters,
                                                    (size, size),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
        self.batchnorm = keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = keras.layers.Dropout(0.5)

    def call(self, x1, x2, training=None):

        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)
        return x

class Generator(keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(32, 4, apply_batchnorm=False)
        self.down2 = Downsample(64, 4)
        self.down3 = Downsample(128, 4)
        self.down4 = Downsample(256, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)


        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(256, 4, apply_dropout=True)
        self.up3 = Upsample(128, 4)
        self.up4 = Upsample(64, 4)
        self.up5 = Upsample(32, 4)

        self.last = keras.layers.Conv2DTranspose(1, (4, 4),
                                                 strides=2,
                                                 padding='same',
                                                 kernel_initializer=initializer)

    def call(self, x, training=None):
        # x shape == (bs, 64, 64, 2)
        x1 = self.down1(x, training=training)  # (bs, 32, 32, 16)  (bs, 16, 16, 16)
        x2 = self.down2(x1, training=training)  # (bs, 16, 16, 32)    (bs, 8, 8, 16)
        x3 = self.down3(x2, training=training)  # (bs, 8, 8, 64)  (bs, 4, 4, 16)
        x4 = self.down4(x3, training=training)  # (bs, 4, 4, 128)  (bs, 2, 2, 16)
        x5 = self.down5(x4, training=training)  # (bs, 2, 2, 128)  (bs, 1, 1, 16)
        x6 = self.down6(x5, training=training)  # (bs, 1, 1, 128)



        u1 = self.up1(x6, x5, training=training)  # (bs, 2, 2, 128)
        u2 = self.up2(u1, x4, training=training)  # (bs, 4, 4, 128)
        u3 = self.up3(u2, x3, training=training)  # (bs, 8, 8, 64)
        u4 = self.up4(u3, x2, training=training)  # (bs, 16, 16, 32)
        u5 = self.up5(u4, x1, training=training)  # (bs, 32, 32, 16)

        # u1 = self.up1(x5, x4, training=training)  # (bs, 2, 2, 128)
        # u2 = self.up2(u1, x3, training=training)  # (bs, 4, 4, 128)
        # u3 = self.up3(u2, x2, training=training)  # (bs, 8, 8, 64)
        # u4 = self.up4(u3, x1, training=training)  # (bs, 16, 16, 32)

        out = self.last(u5)  # (bs, 64, 64, 1)
        # out = self.last(u4)  # (bs, 64, 64, 1)
        # out = tf.nn.tanh(out)

        return out



