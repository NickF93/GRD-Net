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

def build_res_ae(
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
    _, enc_out, _ = rae_obj.gen_encoder(inputs=x, name='encoder', wide=wide)

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
    _, decoder0_residual_output = rae_obj.gen_decoder(inputs=x, last_act=last_act, name='decoder0', wide=wide)

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
    if isinstance(generator_output, (tuple, list)):
        generator_output = generator_output[0]
    generator_model = tf.keras.models.Model(inputs=(generator_input,), outputs=(generator_output, autoencoder_output, latent_space))

    return encoder_model, autencoder_model, generator_model

def build_res_disc(
        img_height: int = 224,
        img_width: Optional[int] = None,
        channels: int = 3,
        init_filters: int = 64,
        use_bias: bool = False,
        initial_padding: int = -1,
        initial_padding_filters: int = -1,
        name: str = 'des_disc',
        wide: int = 1
    ) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:

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
    _, enc_out, _ = rae_obj.gen_encoder(inputs=x, name='encoder', wide=wide)

    x = enc_out

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    discriminator_model = tf.keras.models.Model(inputs=(inputs,), outputs=(enc_out, x), name=f'discriminator_{name}')
    return discriminator_model

def build_res_unet(
        img_height: int = 224,
        img_width: Optional[int] = None,
        channels: int = 3,
        init_filters: int = 64,
        latent_size: int = 128,
        skips: int = 4,
        use_bias: bool = False,
        initial_padding: int = -1,
        initial_padding_filters: int = -1,
        name: str = 'res_unet',
        wide: int = 1,
        residual: bool = True
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
    residual : bool, optional
        If True build a Residual U-Net, else a classi DCNN-based U-Net

    Returns:
    --------
    Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]
        A tuple containing the encoder model, the autoencoder model, and the generator model.
    """
    
    # Ensure image height is provided
    assert img_height is not None, '`img_height` must not be None'

    num_skips: int = int(skips)
    del skips
    
    # If image width is not provided, default it to the height
    if img_width is None:
        img_width = img_height
    
    # Define the input shape of the image
    image_shape = (img_height, img_width, channels)

    # Create the input tensor for the model
    inputs_o = tf.keras.layers.Input(shape=image_shape, name='original_image_input')
    inputs_f = tf.keras.layers.Input(shape=image_shape, name='fake_image_input')

    inputs = (inputs_o, inputs_f)

    xo = inputs_o
    xf = inputs_f

    # If initial padding is specified, apply padding and a convolutional layer
    if initial_padding > 0:
        # Set the filter size for initial padding convolution
        if initial_padding_filters > 0:
            initial_padding_filters = initial_padding_filters
        else:
            initial_padding_filters = channels
        
        # Apply padding followed by a convolution and activation
        xo = tf.keras.layers.ZeroPadding2D(padding=initial_padding, name=f'pre_pad_{name}_o')(xo)
        xo = tf.keras.layers.Conv2D(
            filters=initial_padding_filters, 
            kernel_size=int((initial_padding * 2) + 1), 
            strides=1, 
            padding='valid', 
            use_bias=use_bias, 
            name=f'pre_pad_conv_{name}_o'
        )(xo)
        xo = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'pre_pad_act_{name}_o')(xo)
        
        # Apply padding followed by a convolution and activation
        xf = tf.keras.layers.ZeroPadding2D(padding=initial_padding, name=f'pre_pad_{name}_f')(xf)
        xf = tf.keras.layers.Conv2D(
            filters=initial_padding_filters, 
            kernel_size=int((initial_padding * 2) + 1), 
            strides=1, 
            padding='valid', 
            use_bias=use_bias, 
            name=f'pre_pad_conv_{name}_f'
        )(xf)
        xf = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'pre_pad_act_{name}_f')(xf)

    x = tf.keras.layers.Concatenate(axis=-1, name=f'concat_inputs_{name}')([xo, xf])

    # Create an instance of the ResnetAE class
    rae_obj = ResnetAE(init_features=init_filters, channels=channels, name=f'rae_{name}', net_shape=(2, 2, 2, 2, 2))

    # Generate the encoder model and retrieve its output
    _, enc_out, skips = rae_obj.gen_encoder(inputs=x, name='encoder', wide=wide)

    x = enc_out

    x = tf.keras.layers.Conv2D(latent_size, 4, strides=1, use_bias=False, padding='valid', name=f'encoder0_conv_bottleneck_nz{latent_size}')(x)
    bf_shape = x.get_shape()[1:]
    x = tf.keras.layers.Flatten(name='encoder0_bottleneck_flat')(x)

    latent_space = x

    # Decoder bottleneck
    decoder0_input = x
    encoder0_out_layer = decoder0_input

    # Reshape if the bottleneck is convolutional
    x = tf.keras.layers.Reshape(bf_shape, name='decoder0_bottleneck_reshape')(decoder0_input)
    x = tf.keras.layers.Conv2DTranspose(enc_out.shape[-1], 4, strides=1, use_bias=False, padding='valid', name=f'decoder0_bottleneck_nz{latent_size}')(x)

    # If initial padding was applied, adjust the final activation accordingly
    last_act = True
    if initial_padding > 0:
        last_act = False

    # Generate the decoder model and retrieve its output
    if len(skips) > num_skips and num_skips > 0:
        skips = skips[-num_skips:]
    tmp_skips = []
    for i, s in enumerate(skips):
        _, _, _, c = tf.keras.backend.int_shape(s)
        s = tf.keras.layers.Conv2D(c // 4, 4, strides=1, use_bias=False, padding='valid', name=f'encoder0_conv_bottleneck_nz{latent_size}_{i}')(s)
        bf_shape_s = s.get_shape()[1:]
        s = tf.keras.layers.Flatten(name=f'encoder0_bottleneck_flat_{i}')(s)

        # Reshape if the bottleneck is convolutional
        s = tf.keras.layers.Reshape(bf_shape_s, name=f'decoder0_bottleneck_reshape_{i}')(s)
        s = tf.keras.layers.Conv2DTranspose(c, 4, strides=1, use_bias=False, padding='valid', name=f'decoder0_bottleneck_nz{latent_size}_{i}')(s)
        tmp_skips.append(s)
    skips = tmp_skips
    
    _, decoder0_residual_output = rae_obj.gen_decoder(inputs=x, last_act=last_act, name='decoder0', wide=wide, skips=skips, override_channels=1)

    # If initial padding was applied, remove it from the final output
    if initial_padding > 0:
        post_x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=int((initial_padding * 2) + 1), strides=1, padding='valid', use_bias=use_bias)(decoder0_residual_output)
        post_x = tf.keras.layers.Activation(activation=tf.nn.sigmoid)(post_x)
        post_x = tf.keras.layers.Cropping2D(cropping=initial_padding)(post_x)
        outputs = post_x
    else:
        outputs = decoder0_residual_output

    # Create the full unet model
    unet_model = tf.keras.models.Model(inputs=(inputs,), outputs=(outputs,), name=f'unet_{name}')

    return unet_model
