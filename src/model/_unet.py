"""
Module containing the UNet class for constructing a U-Net model using TensorFlow.
"""

from typing import Callable, List, Optional, Tuple

import tensorflow as tf


class UNet:
    """
    A class to construct a U-Net architecture using TensorFlow.

    Attributes:
        init_features (int): Number of features in the initial convolutional layer.
        bias (bool): Whether to use bias in convolutional layers.
        last_activation (str): Activation function for the output layer.
        name (str): Name of the model.
    """

    def __init__(
        self,
        init_features: int,
        channels: int,
        bias: bool = False,
        bn: bool = True,
        last_activation: str = 'sigmoid',
        name: str = 'unet_ae',
    ):
        """
        Initializes the UNet model with the given parameters.

        Parameters:
            init_features (int): Number of features in the initial convolutional layer.
            channels (int): Number of output channels.
            bias (bool, optional): Whether to use bias in convolutional layers. Defaults to False.
            bn (bool, optional): Whether to use batch normalization. Defaults to True.
            last_activation (str, optional): Activation function for the output layer. Defaults to 'sigmoid'.
            name (str, optional): Name of the model. Defaults to 'unet_ae'.
        """
        self.init_features = init_features
        self.channels = channels
        self.last_activation = last_activation
        self.bias = bias
        self.bn = bn
        self.name = name

    def _get_activation(
        self, act: str, name: Optional[str] = None
    ) -> tf.keras.layers.Layer:
        """
        Returns the corresponding activation layer for the provided activation name.

        Parameters:
            act (str): The activation function to be applied. Supported values:
                'lrelu', 'leakyrelu', 'relu', 'sigmoid', 'tanh', 'swish', 'silu', 'elu', 'gelu'.
            name (Optional[str]): Optional name for the activation layer. Defaults to None.

        Returns:
            tf.keras.layers.Layer: A Keras activation layer corresponding to the specified activation function.

        Raises:
            ValueError: If the provided activation function is not supported.
        """
        act = act.lower().strip()
        if act in ['lrelu', 'leakyrelu']:
            return tf.keras.layers.LeakyReLU(alpha=0.2, name=name)
        elif act == 'relu':
            return tf.keras.layers.ReLU(name=name)
        elif act == 'sigmoid':
            return tf.keras.layers.Activation(tf.nn.sigmoid, name=name)
        elif act == 'tanh':
            return tf.keras.layers.Activation(tf.nn.tanh, name=name)
        elif act == 'swish':
            return tf.keras.layers.Activation(tf.nn.swish, name=name)
        elif act == 'silu':
            return tf.keras.layers.Activation(tf.nn.silu, name=name)
        elif act == 'elu':
            return tf.keras.layers.Activation(tf.nn.elu, name=name)
        elif act == 'gelu':
            return tf.keras.layers.Activation(tf.nn.gelu, name=name)
        else:
            raise ValueError(
                "Activation function must be one of 'lrelu', 'leakyrelu', 'relu', 'sigmoid', "
                "'tanh', 'swish', 'silu', 'elu', or 'gelu'."
            )

    def conv_block(
        self, filters: int, filters1: Optional[int] = None
    ) -> Callable[[tf.Tensor], tf.Tensor]:
        """
        Creates a convolutional block with optional batch normalization and activation.

        Parameters:
            filters (int): Number of filters for the convolutional layers.
            filters1 (Optional[int]): Number of filters for the second convolutional layer. Defaults to None.

        Returns:
            Callable[[tf.Tensor], tf.Tensor]: A function that applies the convolutional block to an input tensor.
        """

        def _block(x: tf.Tensor) -> tf.Tensor:
            """
            Applies the convolutional block to the input tensor.

            Parameters:
                x (tf.Tensor): Input tensor.

            Returns:
                tf.Tensor: Output tensor after applying the convolutional block.
            """
            f = filters
            x = tf.keras.layers.Conv2D(
                f, kernel_size=3, padding='same', use_bias=self.bias
            )(x)
            if self.bn:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            if filters1 is not None:
                f = filters1
            x = tf.keras.layers.Conv2D(
                f, kernel_size=3, padding='same', use_bias=self.bias
            )(x)
            if self.bn:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            return x

        return _block

    def down_block(self, k: int = 2) -> Callable[[tf.Tensor], tf.Tensor]:
        """
        Creates a downsampling block using MaxPooling.

        Parameters:
            k (int, optional): Pool size for MaxPooling. Defaults to 2.

        Returns:
            Callable[[tf.Tensor], tf.Tensor]: A function that applies the downsampling block to an input tensor.
        """

        def _block(x: tf.Tensor) -> tf.Tensor:
            """
            Applies the downsampling block to the input tensor.

            Parameters:
                x (tf.Tensor): Input tensor.

            Returns:
                tf.Tensor: Output tensor after downsampling.
            """
            x = tf.keras.layers.MaxPool2D(pool_size=k)(x)
            return x

        return _block

    def up_block(
        self, k: int, filters: int
    ) -> Callable[[tf.Tensor], tf.Tensor]:
        """
        Creates an upsampling block using UpSampling and convolution.

        Parameters:
            k (int): Upsampling size.
            filters (int): Number of filters for the convolutional layer.

        Returns:
            Callable[[tf.Tensor], tf.Tensor]: A function that applies the upsampling block to an input tensor.
        """

        def _block(x: tf.Tensor) -> tf.Tensor:
            """
            Applies the upsampling block to the input tensor.

            Parameters:
                x (tf.Tensor): Input tensor.

            Returns:
                tf.Tensor: Output tensor after upsampling.
            """
            x = tf.keras.layers.UpSampling2D(
                size=k, interpolation="bilinear"
            )(x)
            x = tf.keras.layers.Conv2D(
                filters, kernel_size=3, padding='same', use_bias=self.bias
            )(x)
            if self.bn:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            return x

        return _block

    def generate_ds_encoder(
        self,
        x: tf.Tensor,
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Generates the downsampling (encoder) part of the U-Net.

        Parameters:
            x (tf.Tensor): Input tensor.

        Returns:
            Tuple[tf.Tensor, List[tf.Tensor]]: The input tensor and a list of tensors from each block.
        """
        inputs = x
        block_outputs: List[tf.Tensor] = []

        # First block
        k = 1
        x = self.conv_block(self.init_features * k)(x)
        block_outputs.append(x)
        x = self.down_block(2)(x)

        # Second block
        k = 2
        x = self.conv_block(self.init_features * k)(x)
        block_outputs.append(x)
        x = self.down_block(2)(x)

        # Third block
        k = 4
        x = self.conv_block(self.init_features * k)(x)
        block_outputs.append(x)
        x = self.down_block(2)(x)

        # Fourth block
        k = 8
        x = self.conv_block(self.init_features * k)(x)
        block_outputs.append(x)
        x = self.down_block(2)(x)

        # Bottleneck
        k = 8
        x = self.conv_block(self.init_features * k)(x)
        block_outputs.append(x)

        return inputs, block_outputs

    def generate_ds_decoder(
        self,
        encoder_outputs: List[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generates the upsampling (decoder) part of the U-Net.

        Parameters:
            encoder_outputs (List[tf.Tensor]): List of tensors from the encoder blocks.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The input tensor and the output tensor of the decoder.
        """
        x = encoder_outputs.pop()
        inputs = x

        # First upsampling block
        k = 8
        k1 = 8
        x = self.up_block(2, self.init_features * k)(x)
        skip_connection = encoder_outputs.pop()
        x = tf.keras.layers.Concatenate(axis=-1)([x, skip_connection])
        x = self.conv_block(self.init_features * k, filters1=self.init_features * k1)(x)

        # Second upsampling block
        k = 4
        k1 = 4
        x = self.up_block(2, self.init_features * k)(x)
        skip_connection = encoder_outputs.pop()
        x = tf.keras.layers.Concatenate(axis=-1)([x, skip_connection])
        x = self.conv_block(self.init_features * k, filters1=self.init_features * k1)(x)

        # Third upsampling block
        k = 2
        k1 = 2
        x = self.up_block(2, self.init_features * k)(x)
        skip_connection = encoder_outputs.pop()
        x = tf.keras.layers.Concatenate(axis=-1)([x, skip_connection])
        x = self.conv_block(self.init_features * k, filters1=self.init_features * k1)(x)

        # Fourth upsampling block
        k = 1
        k1 = 1
        x = self.up_block(2, self.init_features * k)(x)
        skip_connection = encoder_outputs.pop()
        x = tf.keras.layers.Concatenate(axis=-1)([x, skip_connection])
        x = self.conv_block(self.init_features * k, filters1=self.init_features * k1)(x)

        # Output layer
        x = tf.keras.layers.Conv2D(
            self.channels, kernel_size=3, padding='same', use_bias=self.bias
        )(x)
        x = tf.keras.layers.Activation(
            self._get_activation(self.last_activation)
        )(x)
        outputs = x

        return inputs, outputs
