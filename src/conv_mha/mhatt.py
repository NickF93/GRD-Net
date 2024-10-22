from typing import Optional, Union, Tuple, List

import tensorflow as tf

_TupleType = Union[int, Tuple[int, int], List[int]]


def get_act(act: Optional[str], name: Optional[str] = None) -> tf.keras.layers.Layer:
    """
    Returns the corresponding activation layer for the provided activation name.

    Parameters
    ----------
    act : Optional[str]
        The activation function to be applied. Supported values:
        'lrelu', 'leakyrelu', 'relu', 'sigmoid', 'tanh', 'swish',
        'silu', 'elu', 'gelu'. If None, a linear activation is returned.
    name : Optional[str], default None
        Optional name for the activation layer.

    Returns
    -------
    tf.keras.layers.Layer
        A Keras activation layer corresponding to the specified activation function.

    Raises
    ------
    ValueError
        If the provided activation function is not one of the supported values.
    """
    if act is None:
        return tf.keras.layers.Activation('linear', name=name)

    act = act.lower().strip()
    if act in ('lrelu', 'leakyrelu'):
        return tf.keras.layers.LeakyReLU(alpha=0.2, name=name)
    elif act == 'relu':
        return tf.keras.layers.ReLU(name=name)
    elif act == 'sigmoid':
        return tf.keras.layers.Activation('sigmoid', name=name)
    elif act == 'tanh':
        return tf.keras.layers.Activation('tanh', name=name)
    elif act == 'swish':
        return tf.keras.layers.Activation(tf.nn.swish, name=name)
    elif act == 'silu':
        return tf.keras.layers.Activation(tf.nn.silu, name=name)
    elif act == 'elu':
        return tf.keras.layers.Activation('elu', name=name)
    elif act == 'gelu':
        return tf.keras.layers.Activation(tf.nn.gelu, name=name)
    else:
        raise ValueError(
            "act must be one of 'lrelu', 'leakyrelu', 'relu', 'sigmoid', "
            "'tanh', 'swish', 'silu', 'elu', or 'gelu'"
        )


def check_int_tuple(t_el: _TupleType) -> Tuple[int, int]:
    """
    Ensures that the input is a tuple of two integers.

    Parameters
    ----------
    t_el : Union[int, Tuple[int, int], List[int]]
        The element to check and convert.

    Returns
    -------
    Tuple[int, int]
        A tuple of two integers.

    Raises
    ------
    ValueError
        If the input is not an int, tuple, or list, or if the tuple/list does not have length 2.
    """
    if t_el is None:
        raise ValueError('Element must not be None')

    if isinstance(t_el, int):
        ret_t_el = (t_el, t_el)
    elif isinstance(t_el, (tuple, list)):
        ret_t_el = tuple(t_el)
    else:
        raise ValueError('Element must be of type int, tuple, or list')

    if len(ret_t_el) != 2:
        raise ValueError('Element tuple/list must be of length 2')
    if not all(isinstance(item, int) for item in ret_t_el):
        raise ValueError('Element tuple/list elements must be int')

    return ret_t_el


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Custom multi-head attention layer for 2D inputs.

    Parameters
    ----------
    height : int, default 64
        The height of the input.
    width : Optional[int], default None
        The width of the input. If None, it defaults to `height`.
    channels : int, default 3
        The number of channels in the input.
    embed_channels : int, default 128
        The embedding dimension.
    num_heads : int, default 2
        The number of attention heads.
    projections_kernel : Union[int, Tuple[int, int], List[int]], default (3, 3)
        Kernel size for the projection convolutions.
    projections_strides : Union[int, Tuple[int, int], List[int]], default (1, 1)
        Strides for the projection convolutions.
    projections_dilation_rate : Union[int, Tuple[int, int], List[int]], default (1, 1)
        Dilation rate for the projection convolutions.
    projections_padding : str, default 'same'
        Padding type for the projection convolutions, either 'same' or 'valid'.
    projections_use_bias : bool, default False
        Whether to use bias in the projection convolutions.
    projections_activation : Optional[str], default None
        Activation function to use after projection convolutions.
    last_kernel : Union[int, Tuple[int, int], List[int]], default (1, 1)
        Kernel size for the final convolution.
    last_strides : Union[int, Tuple[int, int], List[int]], default (1, 1)
        Strides for the final convolution.
    last_dilation_rate : Union[int, Tuple[int, int], List[int]], default (1, 1)
        Dilation rate for the final convolution.
    last_padding : str, default 'same'
        Padding type for the final convolution, either 'same' or 'valid'.
    last_use_bias : bool, default False
        Whether to use bias in the final convolution.
    last_activation : Optional[str], default None
        Activation function to use after the final convolution.
    last_dropout : Optional[float], default None
        Dropout rate after the final convolution.

    Raises
    ------
    ValueError
        If `embed_channels` is not divisible by `num_heads`.
    """

    def __init__(
        self,
        height: int = 64,
        width: Optional[int] = None,
        channels: int = 3,
        embed_channels: int = 128,
        num_heads: int = 2,
        projections_kernel: _TupleType = (3, 3),
        projections_strides: _TupleType = (1, 1),
        projections_dilation_rate: _TupleType = (1, 1),
        projections_padding: str = 'same',
        projections_use_bias: bool = False,
        projections_activation: Optional[str] = None,
        last_kernel: _TupleType = (1, 1),
        last_strides: _TupleType = (1, 1),
        last_dilation_rate: _TupleType = (1, 1),
        last_padding: str = 'same',
        last_use_bias: bool = False,
        last_activation: Optional[str] = None,
        last_dropout: Optional[float] = None
    ):
        super(MultiHeadAttention, self).__init__()

        if embed_channels % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads."
            )

        # Process and validate convolution parameters
        projections_kernel = check_int_tuple(projections_kernel)
        projections_strides = check_int_tuple(projections_strides)
        projections_dilation_rate = check_int_tuple(projections_dilation_rate)
        projections_padding = projections_padding.strip().lower()
        if projections_padding not in ['valid', 'same']:
            raise ValueError(
                "projections_padding must be 'valid' or 'same'"
            )

        last_kernel = check_int_tuple(last_kernel)
        last_strides = check_int_tuple(last_strides)
        last_dilation_rate = check_int_tuple(last_dilation_rate)
        last_padding = last_padding.strip().lower()
        if last_padding not in ['valid', 'same']:
            raise ValueError(
                "last_padding must be 'valid' or 'same'"
            )

        self.embed_channels = int(embed_channels)
        self.num_heads = int(num_heads)
        self.projection_dim = self.embed_channels // self.num_heads

        self.height = int(height)
        self.width = int(width) if width is not None else self.height
        self.channels = int(channels)

        # Define Q, K, V projection layers
        self.q_proj = self._create_conv_layer(
            filters=self.embed_channels,
            kernel_size=projections_kernel,
            strides=projections_strides,
            dilation_rate=projections_dilation_rate,
            padding=projections_padding,
            use_bias=projections_use_bias
        )
        self.q_act = get_act(projections_activation) if projections_activation else None

        self.k_proj = self._create_conv_layer(
            filters=self.embed_channels,
            kernel_size=projections_kernel,
            strides=projections_strides,
            dilation_rate=projections_dilation_rate,
            padding=projections_padding,
            use_bias=projections_use_bias
        )
        self.k_act = get_act(projections_activation) if projections_activation else None

        self.v_proj = self._create_conv_layer(
            filters=self.embed_channels,
            kernel_size=projections_kernel,
            strides=projections_strides,
            dilation_rate=projections_dilation_rate,
            padding=projections_padding,
            use_bias=projections_use_bias
        )
        self.v_act = get_act(projections_activation) if projections_activation else None

        # Define final projection layer
        self.last_proj = self._create_conv_layer(
            filters=self.embed_channels,
            kernel_size=last_kernel,
            strides=last_strides,
            dilation_rate=last_dilation_rate,
            padding=last_padding,
            use_bias=last_use_bias
        )
        self.last_act = get_act(last_activation) if last_activation else None

        # Define dropout layer if specified
        self.dropout = (
            tf.keras.layers.Dropout(last_dropout)
            if last_dropout is not None and isinstance(last_dropout, float)
            else None
        )

    def _create_conv_layer(
        self,
        filters: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int],
        dilation_rate: Tuple[int, int],
        padding: str,
        use_bias: bool
    ) -> tf.keras.layers.Layer:
        """
        Creates a Conv2D layer with the given parameters.

        Parameters
        ----------
        filters : int
            Number of output filters.
        kernel_size : Tuple[int, int]
            Size of the convolution kernel.
        strides : Tuple[int, int]
            Strides of the convolution.
        dilation_rate : Tuple[int, int]
            Dilation rate for dilated convolution.
        padding : str
            Padding type, 'same' or 'valid'.
        use_bias : bool
            Whether to use bias in the convolution.

        Returns
        -------
        tf.keras.layers.Layer
            The Conv2D layer.
        """
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias
        )

    def _split_heads(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        """
        Split the last dimension into (num_heads, projection_dim).

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch_size, height, width, embed_channels).

        Returns
        -------
        tf.Tensor
            Tensor reshaped to split heads,
            shape (batch_size, num_heads, height, width, projection_dim).
        """
        # x shape: (batch_size, height, width, embed_channels)
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        # Reshape to (batch_size, height, width, num_heads, projection_dim)
        x = tf.reshape(x, (batch_size, height, width, self.num_heads, self.projection_dim))
        # Transpose to (batch_size, num_heads, height, width, projection_dim)
        x = tf.transpose(x, perm=[0, 3, 1, 2, 4])
        return x

    def _get_positional_encoding(
        self,
        height: int,
        width: int,
        channels: int
    ) -> tf.Tensor:
        """
        Generates a simple 2D positional encoding.

        Parameters
        ----------
        height : int
            Height of the input.
        width : int
            Width of the input.
        channels : int
            Number of channels.

        Returns
        -------
        tf.Tensor
            Positional encoding tensor of shape (1, height, width, channels).
        """
        # Create grids of positions with values between -1 and 1
        y = tf.linspace(-1.0, 1.0, height)
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.reshape(y, (height, 1))
        x = tf.reshape(x, (1, width))
        y = tf.tile(y, [1, width])  # shape (height, width)
        x = tf.tile(x, [height, 1])  # shape (height, width)

        # Stack to get (height, width, 2)
        pos_encoding = tf.stack([x, y], axis=-1)  # shape (height, width, 2)

        # Expand to (1, height, width, 2)
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        # If channels > 2, tile or interpolate to match channels
        if channels > 2:
            repeats = channels // 2
            pos_encoding = tf.tile(pos_encoding, [1, 1, 1, repeats])
            remainder = channels % 2
            if remainder != 0:
                # Add additional channels if channels is odd
                pos_encoding_extra = tf.zeros((1, height, width, remainder))
                pos_encoding = tf.concat([pos_encoding, pos_encoding_extra], axis=-1)
        elif channels < 2:
            pos_encoding = pos_encoding[:, :, :, :channels]

        return pos_encoding  # shape (1, height, width, channels)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Forward pass for the multi-head attention layer.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            A tuple containing the query (q), key (k), and value (v) tensors.
            Each tensor should have shape (batch_size, height, width, channels).
        training : Optional[bool], default None
            Training mode indicator.

        Returns
        -------
        tf.Tensor
            Output tensor after applying multi-head attention.
        """
        q, k, v = inputs

        # Linear projections for Q
        q = self.q_proj(q)  # Shape: (batch_size, height, width, embed_channels)
        if self.q_act is not None:
            q = self.q_act(q)

        # Linear projections for K
        k = self.k_proj(k)
        if self.k_act is not None:
            k = self.k_act(k)

        # Linear projections for V
        v = self.v_proj(v)
        if self.v_act is not None:
            v = self.v_act(v)

        # Infer spatial dimensions
        batch_size = tf.shape(q)[0]
        height = tf.shape(q)[1]
        width = tf.shape(q)[2]

        # Add positional encoding to Q and K
        pos_encoding = self._get_positional_encoding(height, width, self.embed_channels)
        q += pos_encoding
        k += pos_encoding

        # Split into multiple heads
        # q, k, v have shape: (batch_size, num_heads, height, width, projection_dim)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Compute attention scores using tf.einsum, keeping height and width axes separate
        # scores shape: (batch_size, num_heads, height_q, width_q, height_k, width_k)
        scores = tf.einsum('bnhwc,bnHWC->bnhwHW', q, k)

        # Scale scores by sqrt(d_k)
        d_k = tf.cast(self.projection_dim, tf.float32)
        scaled_scores = scores / tf.math.sqrt(d_k)

        # Apply softmax over the key height and width axes
        # For numerical stability, subtract the max
        # Compute attention weights
        scaled_scores_reshaped = tf.reshape(
            scaled_scores,
            shape=(-1, self.num_heads, height, width, height * width)
        )
        attention_weights = tf.nn.softmax(scaled_scores_reshaped, axis=-1)
        # Reshape attention weights back to original shape
        attention_weights = tf.reshape(
            attention_weights,
            shape=(-1, self.num_heads, height, width, height, width)
        )

        # Compute context vector
        # context shape: (batch_size, num_heads, height, width, projection_dim)
        context = tf.einsum('bnhwHW,bnHWc->bnhwc', attention_weights, v)

        # Reshape context to (batch_size, height, width, embed_channels)
        context = tf.transpose(context, perm=[0, 2, 3, 1, 4])  # (batch_size, height, width, num_heads, projection_dim)
        context = tf.reshape(context, (batch_size, height, width, self.embed_channels))

        # Final linear projection
        output = self.last_proj(context)
        if self.last_act is not None:
            output = self.last_act(output)

        # Apply dropout if specified
        if self.dropout is not None:
            output = self.dropout(output, training=training)

        return output
