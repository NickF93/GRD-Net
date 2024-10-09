from typing import Optional, Callable
from enum import Enum

import tensorflow as tf

class BottleNeckType(Enum):
    DENSE = 0
    CONVOLUTIONAL = 1

def create_res_ae(
    img_height: int = 224,
    img_width: Optional[int] = None,
    channels: int = 3,
    latent_size: int = 128,
    activation: Callable[[tf.Tensor, Optional[str]], tf.Tensor] = tf.nn.sigmoid,
    bottleneck_type: BottleNeckType = BottleNeckType.CONVOLUTIONAL,
    use_bias: bool = False,
    initial_padding: int = -1,
    initial_padding_filters: int = -1):
    
    assert img_height is not None, '`img_height` must not be None'
    
    if img_width is None:
        img_width = img_height
    
    image_shape = (img_height, img_width, channels)

    inputs = tf.keras.layers.Input(shape=image_shape)

    x = inputs

    if initial_padding > 0:
        if initial_padding_filters > 0:
            initial_padding_filters = initial_padding_filters
        else:
            initial_padding_filters = channels
        pre_x = tf.keras.layers.ZeroPadding2D(padding=initial_padding, name='pre_pad')(x)
        pre_x = tf.keras.layers.Conv2D(filters=initial_padding_filters, kernel_size=int((initial_padding * 2) + 1), strides=1, padding='valid', use_bias=use_bias, name='pre_pad_conv')(pre_x)
        pre_x = tf.keras.layers.LeakyReLU(alpha=0.2, name='pre_pad_act')(pre_x)
