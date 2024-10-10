"""
This module defines functions to create a ResNet-based Autoencoder (ResnetAE) 
with customizable encoder and decoder architectures, bottleneck types (Dense or Convolutional), 
and initial padding layers. The module also defines the `BottleNeckType` enum for specifying
the bottleneck type, and the `create_res_ae` function to generate the autoencoder model, 
encoder model, and a generator model.
"""

from typing import Optional, Tuple, Union
from enum import Enum

import tensorflow as tf
from ._ae import ResnetAE

class BottleNeckType(Enum):
    """
    An enumeration that defines the bottleneck types for the autoencoder's architecture.
    DENSE: Uses a fully connected (dense) bottleneck.
    CONVOLUTIONAL: Uses a convolutional bottleneck.
    """
    DENSE = 0
    CONVOLUTIONAL = 1

def create_res_ae(
        img_height: int = 224,
        img_width: Optional[int] = None,
        channels: int = 3,
        init_filters: int = 64,
        latent_size: int = 128,
        bottleneck_type: BottleNeckType = BottleNeckType.CONVOLUTIONAL,
        use_bias: bool = False,
        initial_padding: int = -1,
        initial_padding_filters: int = -1,
        name: str = 'res_ae',
        wide: int = 1
    ) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """
    Creates a ResNet-based Autoencoder (ResnetAE) with customizable bottleneck type 
    (either Dense or Convolutional), initial padding, and filter sizes. 
    The function returns the encoder, autoencoder, and generator models.

    Parameters:
    -----------
    img_height : int, optional
        The height of the input image. Default is 224.
    img_width : Optional[int], optional
        The width of the input image. If None, it defaults to `img_height`. Default is None.
    channels : int, optional
        The number of input channels (e.g., 3 for RGB). Default is 3.
    init_filters : int, optional
        The number of initial filters in the convolutional layers. Default is 64.
    latent_size : int, optional
        The size of the latent space representation. Default is 128.
    bottleneck_type : BottleNeckType, optional
        The type of bottleneck to use (Dense or Convolutional). Default is BottleNeckType.CONVOLUTIONAL.
    use_bias : bool, optional
        Whether to use bias in the convolutional layers. Default is False.
    initial_padding : int, optional
        Size of the initial padding. If less than 0, no padding is applied. Default is -1.
    initial_padding_filters : int, optional
        Number of filters to apply after initial padding. If less than 0, defaults to the number of input channels. Default is -1.
    name : str, optional
        Name of the model. Default is 'res_ae'.
    wide : int, optional
        The widening factor for the residual network. Default is 1.

    Returns:
    --------
    Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]
        A tuple containing the encoder model, the autoencoder model, and the generator model.
    """
    
    # Ensure image height is provided
    assert img_height is not None, '`img_height` must not be None'
    
    # If image width is not provided, default it to the height
    if img_width is None:
        img_width = img_height
    
    # Define the input shape of the image
    image_shape = (img_height, img_width, channels)

    # Create the input tensor for the model
    inputs = tf.keras.layers.Input(shape=image_shape)

    x = inputs

    # If initial padding is specified, apply padding and a convolutional layer
    if initial_padding > 0:
        # Set the filter size for initial padding convolution
        if initial_padding_filters > 0:
            initial_padding_filters = initial_padding_filters
        else:
            initial_padding_filters = channels
        
        # Apply padding followed by a convolution and activation
        x = tf.keras.layers.ZeroPadding2D(padding=initial_padding, name=f'pre_pad_{name}')(x)
        x = tf.keras.layers.Conv2D(
            filters=initial_padding_filters, 
            kernel_size=int((initial_padding * 2) + 1), 
            strides=1, 
            padding='valid', 
            use_bias=use_bias, 
            name=f'pre_pad_conv_{name}'
        )(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'pre_pad_act_{name}')(x)

    # Create an instance of the ResnetAE class
    rae_obj = ResnetAE(init_features=init_filters, channels=channels, name=f'rae_{name}', net_shape=(2, 2, 2, 2, 2))

    # Generate the encoder model and retrieve its output
    _, _, enc_out = rae_obj.gen_encoder(inputs=x, name='encoder', wide=wide)

    x = enc_out

    # If using a dense bottleneck, flatten the output
    if bottleneck_type == BottleNeckType.DENSE:
        bf_shape = x.get_shape()[1:]
        x = tf.keras.layers.Flatten(name='encoder0_bottleneck_flat')(x)

    # Encoder bottleneck
    if bottleneck_type == BottleNeckType.DENSE:
        x = tf.keras.layers.Dense(latent_size, name=f'encoder0_dense_bottleneck_nz{latent_size}')(x)
    else:
        x = tf.keras.layers.Conv2D(latent_size, 4, strides=1, use_bias=False, padding='valid', name=f'encoder0_conv_bottleneck_nz{latent_size}')(x)
        bf_shape = x.get_shape()[1:]
        x = tf.keras.layers.Flatten(name='encoder0_bottleneck_flat')(x)

    latent_space = x

    # Create the encoder model
    encoder_model = tf.keras.models.Model(inputs=(inputs,), outputs=(latent_space,), name=f'encder_model_{name}')

    # Decoder bottleneck
    decoder0_input = x
    encoder0_out_layer = decoder0_input

    if bottleneck_type == BottleNeckType.DENSE:
        x = decoder0_input
    else:
        # Reshape if the bottleneck is convolutional
        x = tf.keras.layers.Reshape(bf_shape, name='decoder0_bottleneck_reshape')(decoder0_input)
        x = tf.keras.layers.Conv2DTranspose(enc_out.shape[-1], 4, strides=1, use_bias=False, padding='valid', name=f'decoder0_bottleneck_nz{latent_size}')(x)

    # If using a dense bottleneck, reshape the output
    if bottleneck_type == BottleNeckType.DENSE:
        prod_shape = tf.math.reduce_prod(bf_shape)
        x = tf.keras.layers.Dense(prod_shape, name=f'decoder0_preflatten_dense_{prod_shape.numpy()}')(x)
        x = tf.keras.layers.Reshape(bf_shape, name='decoder0_bottleneck_reshape')(x)

    # If initial padding was applied, adjust the final activation accordingly
    last_act = True
    if initial_padding > 0:
        last_act = False

    # Generate the decoder model and retrieve its output
    _, _, decoder0_residual_output = rae_obj.gen_decoder(inputs=x, last_act=last_act, name='decoder0', wide=wide)

    # If initial padding was applied, remove it from the final output
    if initial_padding > 0:
        post_x = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=int((initial_padding * 2) + 1), strides=1, padding='valid', use_bias=use_bias)(decoder0_residual_output)
        post_x = tf.keras.layers.Activation(activation=tf.nn.sigmoid)(post_x)
        post_x = tf.keras.layers.Cropping2D(cropping=initial_padding)(post_x)
        outputs = post_x
    else:
        outputs = decoder0_residual_output

    # Create the full autoencoder model
    autencoder_model = tf.keras.models.Model(inputs=(inputs,), outputs=(outputs, latent_space), name=f'autoencoder_{name}')

    # Create the generator model by chaining the encoder and autoencoder models
    generator_input = tf.keras.layers.Input(shape=image_shape)
    autoencoder_output, latent_space = autencoder_model(generator_input)
    generator_output = encoder_model(autoencoder_output)
    generator_model = tf.keras.models.Model(inputs=(generator_input,), outputs=(generator_output, autoencoder_output, latent_space))

    return encoder_model, autencoder_model, generator_model
