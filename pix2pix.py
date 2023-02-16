from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, BatchNormalization, LeakyReLU
from tensorflow import keras
import tensorflow as tf

def unet(img_shape,generator_outputs_channels,ngf=64):

    i = keras.Input(shape=img_shape, dtype=tf.float32)
    # [n, 640, 360, n] -> [n, 320, 180, 64]
    l1 = Conv2D(ngf, (4, 4), strides=(2, 2), padding='same', input_shape=img_shape)(i)
    l1 = BatchNormalization()(l1)
    l1 = LeakyReLU()(l1)

    l2 = Conv2D(ngf*2, (4, 4), strides=(2, 2), padding='same')(l1)
    l2 = BatchNormalization()(l2)
    l2 = LeakyReLU()(l2)

    l3 = Conv2D(ngf*4, (4, 4), strides=(2, 2), padding='same')(l2)
    l3 = BatchNormalization()(l3)
    l3 = LeakyReLU()(l3)

    l4 = Conv2D(ngf*8, (4, 4), strides=(2, 2), padding='same')(l3)
    l4 = BatchNormalization()(l4)
    l4 = LeakyReLU()(l4)

    l5 = Conv2D(ngf*8, (4, 4), strides=(2, 2), padding='same')(l4)
    l5 = BatchNormalization()(l5)
    l5 = LeakyReLU()(l5)

    l6 = Conv2D(ngf*8, (4, 4), strides=(2, 2), padding='same')(l5)
    l6 = BatchNormalization()(l6)
    l6 = LeakyReLU()(l6)

    l7 = Conv2D(ngf*8, (4, 4), strides=(2, 2), padding='same')(l6)
    l7 = BatchNormalization()(l7)
    l7 = LeakyReLU()(l7)

    l8 = Conv2D(ngf*8, (4, 4), strides=(2, 2), padding='same')(l7)
    l8 = BatchNormalization()(l8)
    l8 = LeakyReLU()(l8)

    u1 = Conv2DTranspose(ngf*8, (4, 4), strides=(2, 2), padding='same')(l8)
    u1 = BatchNormalization()(u1)
    u1 = ReLU()(u1)
    u1 = tf.concat((u1, l7), axis=3)    

    u2 = Conv2DTranspose(ngf*8, (4, 4), strides=(2, 2), padding='same')(u1)
    u2 = BatchNormalization()(u2)
    u2 = ReLU()(u2)
    u2 = tf.concat((u2, l6), axis=3)    

    u3 = Conv2DTranspose(ngf*8, (4, 4), strides=(2, 2), padding='same')(u2)
    u3 = BatchNormalization()(u3)
    u3 = ReLU()(u3)
    u3 = tf.concat((u3, l5), axis=3)    

    u4 = Conv2DTranspose(ngf*8, (4, 4), strides=(2, 2), padding='same')(u3)
    u4 = BatchNormalization()(u4)
    u4 = ReLU()(u4)
    u4 = tf.concat((u4, l4), axis=3)  

    u5 = Conv2DTranspose(ngf*4, (4, 4), strides=(2, 2), padding='same')(u4)
    u5 = BatchNormalization()(u5)
    u5 = ReLU()(u5)
    u5 = tf.concat((u5, l3), axis=3)  

    u6 = Conv2DTranspose(ngf*2, (4, 4), strides=(2, 2), padding='same')(u5)
    u6 = BatchNormalization()(u6)
    u6 = ReLU()(u6)
    u6 = tf.concat((u6, l2), axis=3)   

    u7 = Conv2DTranspose(ngf*1, (4, 4), strides=(2, 2), padding='same')(u6)
    u7 = BatchNormalization()(u7)
    u7 = ReLU()(u7)
    u7 = tf.concat((u7, l1), axis=3)   

    u8 = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(u7)
    u8 = BatchNormalization()(u8)
    u8 = ReLU()(u8)
    u8 = tf.concat((u8, i), axis=3)     

    # -> [n, 640,360,3]
    o = Conv2D(3, (4, 4), padding='same', activation=keras.activations.tanh)(u8)

    unet = keras.Model(i, o, name="unet")

    tf.keras.utils.plot_model(unet, to_file='generatorWithShape.png',show_shapes=True)
    unet.summary()
    return unet

def patchNet(ndf=64):
    input_img = keras.Input(shape=(512,512,5))
    generated_img = keras.Input(shape=(512,512,3))
    concat_img = tf.concat((input_img, generated_img), axis=-1)
    model = keras.Sequential()
    model.add(Conv2D(ndf, (4, 4), strides=(2, 2), padding='same', input_shape=(512,512,5+3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(ndf*2, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(ndf*4, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(ndf*8, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(1, (4, 4)))

    o = tf.squeeze(model(concat_img), axis=-1)

    patch_gan = keras.Model([input_img, generated_img], o, name="patch_gan")

    tf.keras.utils.plot_model(model, to_file='discriminatorWithShape.png',show_shapes=True)
    model.summary()
    return patch_gan

if (__name__ == '__main__'):
    imageshape=(512,512,5)
    unet(imageshape,3)
    patchNet()
