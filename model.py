import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

#每一层的数目
layernums=[64, 128, 256, 512, 1024]

#每一层的操作
def block(newtuple, n):
    conv = SeparableConv2D(layernums[n], 3, activation='elu', padding='same', kernel_initializer='he_normal')(newtuple)
    conv = SeparableConv2D(layernums[n], 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv)
    return conv

#残差路径
def ResPath(filters, length, inp):
    shortcut = inp
    shortcut = SeparableConv2D(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = SeparableConv2D(inp, filters, 3, 3, activation='elu', padding='same')

    out = add([shortcut, out])
    out = Activation('elu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = SeparableConv2D(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = SeparableConv2D(out, filters, 3, 3, activation='elu', padding='same')

        out = add([shortcut, out])
        out = Activation('elu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


#此处为3层unet，实验时进行层数调整
def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = block(inputs, 0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = ResPath(layernums[0], 4, conv1)     #建立残差路径

    conv2 = block(pool1, 1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = ResPath(layernums[1], 4, conv2)

    conv3 = block(pool2, 2)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    pool3 = ResPath(layernums[2], 4, pool3)

    conv4 = block(pool3, 3)
    drop4 = Dropout(0.5)(conv4)

    #每层采样
    up5 = SeparableConv2D(layernums[2], 2, activation='elu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = block(merge5, 2)

    up6 = SeparableConv2D(layernums[1], 2, activation='elu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = block(merge6, 1)

    up7 = SeparableConv2D(layernums[0], 2, activation='elu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = block(merge7, 0)
    conv7 = SeparableConv2D(2, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = SeparableConv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(input=inputs, output=conv8)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
