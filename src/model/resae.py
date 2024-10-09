from typing import Optional, Callable
from enum import Enum

import tensorflow as tf

from ._ae import ResnetAE

class BottleNeckType(Enum):
    DENSE = 0
    CONVOLUTIONAL = 1

def create_res_ae(
    img_height: int = 224,
    img_width: Optional[int] = None,
    channels: int = 3,
    init_filters: int = 64,
    latent_size: int = 128,
    activation: Callable[[tf.Tensor, Optional[str]], tf.Tensor] = tf.nn.sigmoid,
    bottleneck_type: BottleNeckType = BottleNeckType.CONVOLUTIONAL,
    use_bias: bool = False,
    initial_padding: int = -1,
    initial_padding_filters: int = -1,
    name: str = 'res_ae'):
    
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
        x = tf.keras.layers.ZeroPadding2D(padding=initial_padding, name=f'pre_pad_{name}')(x)
        x = tf.keras.layers.Conv2D(filters=initial_padding_filters, kernel_size=int((initial_padding * 2) + 1), strides=1, padding='valid', use_bias=use_bias, name=f'pre_pad_conv_{name}')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'pre_pad_act_{name}')(x)

    rae_obj = ResnetAE(init_features=init_filters, channels=channels, name=f'rae_{name}', net_shape=(2, 2, 2, 2, 2, 2))

    _, _, enc_out = rae_obj.gen_encoder(inputs=x, name='encoder')

    x = enc_out

    #region [flatten if dense]
    if bottleneck_type == BottleNeckType.DENSE:
        bf_shape = x.get_shape()[1:]
        x = tf.keras.layers.Flatten(name='encoder0_bottleneck_flat')(x)
    #endregion [flatten if dense]

    #region [bottleneck]
    #region [encoder bottleneck]
    if bottleneck_type == BottleNeckType.DENSE:
        x = tf.keras.layers.Dense(latent_size, name='encoder0_dense_bottleneck_nz{}'.format(latent_size))(x)
    else:
        x = tf.keras.layers.Conv2D(latent_size, 4, strides=1, use_bias=False, padding='valid', name='encoder0_conv_bottleneck_nz{}'.format(latent_size))(x)
        bf_shape = x.get_shape()[1:]
        x = tf.keras.layers.Flatten(name='encoder0_bottleneck_flat')(x)

    #endregion [encoder bottleneck]

    latent_space = x

    encoder_model = tf.keras.models.Model(inputs=(inputs,), outputs=(latent_space,), name=f'encder_model_{name}')

    #region [decoder bottleneck]
    decoder0_input = x
    encoder0_out_layer = decoder0_input

    if bottleneck_type == BottleNeckType.DENSE:
      x = decoder0_input
    else:
      x = tf.keras.layers.Reshape(bf_shape, name='decoder0_bottleneck_reshape')(decoder0_input)
      x = tf.keras.layers.Conv2DTranspose(enc_out.shape[-1], 4, strides=1, use_bias=False, padding='valid', name='decoder0_bottleneck_nz{}'.format(latent_size))(x)
    #endregion [decoder bottleneck]
    #endregion [bottleneck]

    #region [reshape if dense]
    if bottleneck_type == BottleNeckType.DENSE:
        prod_shape = tf.math.reduce_prod(bf_shape)
        x = tf.keras.layers.Dense(prod_shape, name='decoder0_preflatten_dense_{}'.format(prod_shape.numpy()))(x)
        x = tf.keras.layers.Reshape(bf_shape, name='decoder0_bottleneck_reshape')(x)
    #endregion [reshape if dense]

    last_act = True
    if initial_padding > 0:
        last_act = False

    _, _, decoder0_residual_output = rae_obj.gen_decoder(inputs=x, last_act=last_act, name='decoder0')

    if initial_padding > 0:
        post_x = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=int((initial_padding * 2) + 1), strides=1, padding='valid', use_bias=use_bias)(decoder0_residual_output)
        post_x = tf.keras.layers.Activation(activation=tf.nn.sigmoid)(post_x)
        post_x = tf.keras.layers.Cropping2D(cropping=initial_padding)(post_x)
        outputs = post_x
    else:
        outputs = decoder0_residual_output

    autencoder_model = tf.keras.models.Model(inputs=(inputs,), outputs=(outputs, latent_space), name=f'autoencoder_{name}')

    generator_input = tf.keras.layers.Input(shape=image_shape)

    autoencoder_output, latent_space = autencoder_model(generator_input)
    generator_output = encoder_model(autoencoder_output)

    generator_model = tf.keras.models.Model(inputs=(generator_input,), outputs=(generator_output, autoencoder_output, latent_space))

    return encoder_model, autencoder_model, generator_model
