"""
This module defines a ResNet-based Autoencoder (ResnetAE) class with customizable
encoder and decoder architectures. The class supports various activation functions,
batch normalization, and the construction of residual blocks for both the encoder
and decoder.

The ResnetAE class offers methods to generate the encoder (`gen_encoder`) and decoder
(`gen_decoder`), along with support for flexible residual block creation, including
the option to use a wide residual network.
"""

from typing import Optional, Tuple, Union, List
import tensorflow as tf

from ..conv_mha import ConvMultiHeadAttention


def _attention_res_block(
        x: tf.Tensor,
        use_bias: bool,
        name: str
) -> tf.Tensor:
    x_skip = x
    x_skip = tf.keras.layers.MaxPool2D((2, 2))(x_skip)

    # Conv Attention
    mhatt = ConvMultiHeadAttention(
        height=int(x.shape[1]),
        width=int(x.shape[2]),
        channels=int(x.shape[3]),
        embed_channels=64,
        num_heads=4,
        projections_kernel=(3, 3),
        projections_strides=(2, 2),
        projections_dilation_rate=(1, 1),
        projections_padding='same',
        projections_use_bias=use_bias,
        projections_activation=None,
        last_kernel=(1, 1),
        last_strides=(1, 1),
        last_dilation_rate=(1, 1),
        last_padding='same',
        last_use_bias=use_bias,
        last_activation=None,
        last_dropout=None
    )
    x = mhatt((x, x, x))
    x = x + x_skip
    x_skip = x
    x = tf.keras.layers.Conv2D(
            filters=128, 
            kernel_size=(1, 1), 
            strides=1, 
            padding='same', 
            use_bias=use_bias, 
            name=f'conv_trans_0_{name}'
    )(x)
    x = tf.keras.layers.Activation('gelu')(x)
    x = tf.keras.layers.Conv2D(
            filters=64, 
            kernel_size=(1, 1), 
            strides=1, 
            padding='same', 
            use_bias=use_bias, 
            name=f'conv_trans_1_{name}'
    )(x)
    x = x + x_skip
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'attention_att_{name}')(x)
    return x


class ResnetAE:
    """
    A ResNet-based Autoencoder class. This class implements a basic
    ResNet-based autoencoder architecture with customizable parameters for
    the encoder and decoder activations, network shape, and more.

    Attributes:
    -----------
    _if : int
        The number of initial features for the network.
    _channels : int
        The number of input channels (e.g., RGB = 3).
    _bias : bool
        Whether to use biases in the convolutional layers.
    _bn : bool
        Whether to apply batch normalization across all layers.
    _flbn : bool
        Whether to apply batch normalization on the first and last layers.
    _iks : int
        The kernel size for the input convolutional layer.
    _ks : int
        The kernel size for the intermediate layers.
    _enc_act : str
        The activation function for the encoder.
    _dec_act : str
        The activation function for the decoder.
    _lst_act : str
        The activation function for the final layer.
    _net_shape : Tuple[int, ...]
        The shape of the network (depth and feature dimensions).
    name : str
        The name of the model.
    """

    def __init__(
        self,
        init_features: int,
        channels: int,
        bias: bool = False,
        first_last_bn: bool = False,
        bn: bool = False,
        input_kernel_size: int = 4,
        kernel_size: int = 3,
        encoder_act: str = 'lrelu',
        decoder_act: str = 'relu',
        last_activation: str = 'sigmoid',
        net_shape: Tuple[int, ...] = (2, 2, 2, 2),
        name: str = 'resnet_ae'
    ):
        """
        Initializes the ResnetAE class with customizable hyperparameters.

        Parameters:
        -----------
        init_features : int
            Number of initial features for the network.
        channels : int
            Number of input channels (e.g., 3 for RGB images).
        bias : bool, optional
            Whether to use biases in the convolutional layers (default is False).
        first_last_bn : bool, optional
            Whether to apply batch normalization on the first and last layers (default is False).
        bn : bool, optional
            Whether to apply batch normalization across all layers (default is False).
        input_kernel_size : int, optional
            Kernel size for the input convolutional layer (default is 4).
        kernel_size : int, optional
            Kernel size for the intermediate convolutional layers (default is 3).
        encoder_act : str, optional
            Activation function for the encoder. Supported values: 'lrelu', 'relu', 'sigmoid', 'tanh', 'swish', 'silu', 'elu', 'gelu' (default is 'lrelu').
        decoder_act : str, optional
            Activation function for the decoder. Supported values: same as encoder_act (default is 'relu').
        last_activation : str, optional
            Activation function for the final layer (default is 'sigmoid').
        net_shape : Tuple[int, ...], optional
            Shape of the network, defined as a tuple of integers for network depth (default is (2, 2, 2, 2)).
        name : str, optional
            Name of the model (default is 'resnet_ae').
        """
        self._if = init_features
        self._channels = channels
        self._bias = bias
        self._bn = bn
        self._flbn = first_last_bn
        self._iks = input_kernel_size
        self._ks = kernel_size
        self._enc_act = encoder_act
        self._dec_act = decoder_act
        self._lst_act = last_activation
        self._net_shape = net_shape
        self.name = str(name)

    def _get_act(self, act: str, name: Optional[str] = None) -> tf.keras.layers.Layer:
        """
        Returns the corresponding activation layer for the provided activation name.

        Parameters:
        -----------
        act : str
            The activation function to be applied. Supported values:
            'lrelu', 'leakyrelu', 'relu', 'sigmoid', 'tanh', 'swish', 'silu', 'elu', 'gelu'.
        name : str, optional
            Optional name for the activation layer (default is None).

        Returns:
        --------
        tf.keras.layers.Layer
            A Keras activation layer corresponding to the specified activation function.

        Raises:
        -------
        ValueError
            If the provided activation function is not one of the supported values.
        """
        act = act.lower().strip()
        if act == 'lrelu' or act == 'leakyrelu':
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
            raise ValueError("act must be one of 'lrelu', 'leakyrelu', 'relu', 'sigmoid', 'tanh', 'swish', 'silu', 'elu', 'gelu'")

    def _res_block(
        self,
        inputs: tf.Tensor,
        filters: int,
        stage: int,
        block: int,
        strides: int,
        cut: str,
        encoder: bool,
        bias: bool,
        act: str,
        name: str
    ) -> tf.Tensor:
        """
        Builds a residual block, either for the encoder or the decoder, with support for 
        'pre' or 'post' cut (shortcut connection).

        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor to the residual block.
        filters : int
            The number of filters for the convolutional layers.
        stage : int
            The stage index (used for naming layers).
        block : int
            The block index (used for naming layers).
        strides : int
            Strides to be applied to the convolutional layers.
        cut : str
            Type of shortcut connection, either 'pre' or 'post'. 
            'pre' connects the input directly, while 'post' processes the input first.
        encoder : bool
            Whether this block is for the encoder (True) or decoder (False).
        bias : bool
            Whether to use biases in the convolutional layers.
        act : str
            Activation function to use. Supported activations include: 'lrelu', 'relu', 'sigmoid', 'tanh', 'swish', 'silu', 'elu', 'gelu'.
        name : str
            Base name for the layers in this block (used for creating layer names).

        Returns:
        --------
        tf.Tensor
            Output tensor of the residual block.

        Raises:
        -------
        ValueError
            If `cut` is not one of ['pre', 'post'].
        """
        x = inputs

        # Define shortcut (cut) connection
        if cut == 'pre':
            # Pre-cut: the shortcut is directly the input
            shortcut = x
        elif cut == 'post':
            # Post-cut: process the shortcut with a convolution or transpose convolution
            if encoder:
                shortcut = tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=strides,
                    use_bias=bias,
                    padding='same',
                    name=f"{name}_stage{stage}_block{block}_scut_conv0"
                )(x)
            else:
                if strides == 1:
                    shortcut = tf.keras.layers.Conv2DTranspose(
                        filters=filters,
                        kernel_size=1,
                        strides=strides,
                        use_bias=bias,
                        padding='same',
                        name=f"{name}_stage{stage}_block{block}_scut_conv0"
                    )(x)
                else:
                    shortcut = tf.keras.layers.Conv2DTranspose(
                        filters=filters // 2,
                        kernel_size=1,
                        strides=strides,
                        use_bias=bias,
                        padding='same',
                        name=f"{name}_stage{stage}_block{block}_scut_conv0"
                    )(x)
            if self._bn:
                shortcut = tf.keras.layers.BatchNormalization(
                    name=f"{name}_stage{stage}_block{block}_scut_bn1"
                )(shortcut)
            
            shortcut = self._get_act(
                act=act, 
                name=f"{name}_stage{stage}_block{block}_scut_act_{act}1"
            )(shortcut)
        else:
            raise ValueError('Cut type must be one of ["pre", "post"].')

        # Sub-block 1: apply convolution or transposed convolution
        if encoder:
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=self._ks,
                strides=strides,
                use_bias=bias,
                padding='same',
                name=f"{name}_stage{stage}_block{block}_conv0"
            )(x)
        else:
            x = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=self._ks,
                strides=1,
                use_bias=bias,
                padding='same',
                name=f"{name}_stage{stage}_block{block}_conv0"
            )(x)

        if self._bn:
            x = tf.keras.layers.BatchNormalization(
                name=f"{name}_stage{stage}_block{block}_bn0"
            )(x)
        
        x = self._get_act(
            act=act, 
            name=f"{name}_stage{stage}_block{block}_act_{act}0"
        )(x)

        # Sub-block 2: apply convolution or transposed convolution
        if encoder:
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=self._ks,
                strides=1,
                use_bias=bias,
                padding='same',
                name=f"{name}_stage{stage}_block{block}_conv1"
            )(x)
        else:
            if cut == 'post' and strides != 1:
                filters //= 2  # Adjust filter size if strides were used in post-cut
            x = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=self._ks,
                strides=strides,
                use_bias=bias,
                padding='same',
                name=f"{name}_stage{stage}_block{block}_conv1"
            )(x)

        if self._bn:
            x = tf.keras.layers.BatchNormalization(
                name=f"{name}_stage{stage}_block{block}_bn1"
            )(x)

        x = self._get_act(
            act=act, 
            name=f"{name}_stage{stage}_block{block}_act_{act}1"
        )(x)

        # Add shortcut connection
        x = tf.keras.layers.Add(name=f"{name}_stage{stage}_block{block}_add0")([x, shortcut])

        return x

    def gen_encoder(
        self,
        inputs: tf.Tensor,
        name: Union[str, int],
        wide: int = 1,
        attention: bool = False
    ) -> Tuple[tf.keras.models.Model, tf.Tensor, tf.Tensor]:
        """
        Generates the encoder part of the ResNet-based autoencoder.

        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor to the encoder.
        name : Union[str, int]
            Name or identifier for the encoder.
        wide : int
            Wide factor.

        Returns:
        --------
        Tuple[tf.keras.models.Model, tf.Tensor, tf.Tensor]
            A tuple containing the encoder model, the input tensor, and the output tensor.
        """
        name = str(self.name + '_' + str(name))
        
        assert wide > 0, '`wide` factor must be greater that 0'

        skips: List[tf.Tensor] = []

        # Initial input layer
        filters = self._if
        x = inputs
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=self._iks,
            strides=1,
            padding='same',
            use_bias=self._bias,
            name=name + '_in_conv0'
        )(x)

        # Optional batch normalization on the first layer
        if self._flbn:
            x = tf.keras.layers.BatchNormalization(name=name + '_in_bn0')(x)

        # Apply the encoder activation function
        x = self._get_act(act=self._enc_act, name=name + '_in_act_' + self._enc_act + '0')(x)

        if attention:
            x = _attention_res_block(x, use_bias=False, name=name)

        # Loop over stages and blocks in the network
        for stage, blocks in enumerate(self._net_shape):
            for block in range(blocks):
                f = filters * (2 ** stage)

                if stage == 0 and block == 0:
                    # No downsampling for the first block in the first stage
                    x = self._res_block(
                        inputs=x,
                        filters=int(f * wide),
                        stage=stage,
                        block=block,
                        strides=1,
                        cut='post',
                        encoder=True,
                        bias=self._bias,
                        act=self._enc_act,
                        name=name
                    )
                elif block == 0:
                    # Downsampling for the first block of subsequent stages
                    x = self._res_block(
                        inputs=x,
                        filters=int(f * wide),
                        stage=stage,
                        block=block,
                        strides=2,
                        cut='post',
                        encoder=True,
                        bias=self._bias,
                        act=self._enc_act,
                        name=name
                    )
                else:
                    # Regular blocks with no downsampling
                    x = self._res_block(
                        inputs=x,
                        filters=int(f * wide),
                        stage=stage,
                        block=block,
                        strides=1,
                        cut='pre',
                        encoder=True,
                        bias=self._bias,
                        act=self._enc_act,
                        name=name
                    )
            skips.append(x)

        # Return the encoder model, inputs, and output tensor
        return inputs, x, skips

    def gen_decoder(
        self,
        inputs: tf.Tensor,
        last_act: bool,
        name: Union[str, int],
        wide: int = 1,
        skips: Optional[Tuple[tf.Tensor, ...]] = None,
        override_channels: Optional[int] = None
    ) -> Tuple[tf.keras.models.Model, tf.Tensor, tf.Tensor]:
        """
        Generates the decoder part of the ResNet-based autoencoder.

        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor to the decoder.
        last_act : bool
            Whether to apply the last activation function.
        name : Union[str, int]
            Name or identifier for the decoder.
        wide : int
            Wide factor.
        skips : Optional[Tupe[tf.Tensor, ...]]
            A tuple of skips connections from Encoder.
        override_channels : Optional[int]
            If is None use the input channels, else use the number of channels provided as integer.

        Returns:
        --------
        Tuple[tf.keras.models.Model, tf.Tensor, tf.Tensor]
            A tuple containing the decoder model, the input tensor, and the output tensor.
        """
        name = str(self.name + '_' + str(name))
        
        assert wide > 0, '`wide` factor must be greater that 0'
        
        filters = self._if
        x = inputs

        last_stage = len(self._net_shape)

        # Loop over stages and blocks in the reverse order for decoding
        for stage, blocks in enumerate(self._net_shape):
            if skips is not None and (isinstance(skips, tuple) or isinstance(skips, list)) and len(skips) > 0:
                _, _, _, c = tf.keras.backend.int_shape(x)
                x = tf.keras.layers.Concatenate(axis=-1)([x, skips.pop()])
                x = tf.keras.layers.Conv2DTranspose(
                    filters=c,
                    kernel_size=1,
                    strides=1,
                    use_bias=self._bias,
                    padding='same',
                    name=f"{name}_stage{stage}_aggregator0"
                )(x)

            for block in range(blocks):
                f = filters * (2 ** ((last_stage - 1) - stage))

                if stage == (last_stage - 1) and block == (blocks - 1):
                    # Final block of the decoder
                    x = self._res_block(
                        inputs=x,
                        filters=int(f * wide),
                        stage=stage,
                        block=block,
                        strides=1,
                        cut='post',
                        encoder=False,
                        bias=self._bias,
                        act=self._dec_act,
                        name=name
                    )
                elif block == (blocks - 1):
                    # Last block of each stage (with downsampling)
                    x = self._res_block(
                        inputs=x,
                        filters=int(f * wide),
                        stage=stage,
                        block=block,
                        strides=2,
                        cut='post',
                        encoder=False,
                        bias=self._bias,
                        act=self._dec_act,
                        name=name
                    )
                else:
                    # Regular blocks with no downsampling
                    x = self._res_block(
                        inputs=x,
                        filters=int(f * wide),
                        stage=stage,
                        block=block,
                        strides=1,
                        cut='pre',
                        encoder=False,
                        bias=self._bias,
                        act=self._dec_act,
                        name=name
                    )

        # Final output layer
        x = tf.keras.layers.Conv2DTranspose(
            filters=self._channels if override_channels is None else int(override_channels),
            kernel_size=self._iks,
            strides=1,
            padding='same',
            use_bias=self._bias,
            name=name + '_in_conv0'
        )(x)

        # Optional batch normalization on the final layer
        if self._flbn:
            x = tf.keras.layers.BatchNormalization(name=name + '_in_bn0')(x)

        # Apply the last activation function if specified, otherwise use LeakyReLU
        if last_act:
            x = self._get_act(act=self._lst_act, name=name + '_in_act_' + self._lst_act + '0')(x)
        else:
            x = self._get_act(act='lrelu', name=name + '_in_act_last_lrelu0')(x)

        # Return the decoder model, inputs, and output tensor
        return inputs, x
