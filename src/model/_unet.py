from typing import Optional, Tuple, Union, List
import tensorflow as tf

class UNet:
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
        name: str = 'unet_ae'
    ):
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
    

    def conv_block(self, filters, filters1=None):
        def _f(x):
            f = filters
            x = tf.keras.layers.Conv2D(f, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            if filters1 is not None:
                f = filters1
            x = tf.keras.layers.Conv2D(f, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            return x
        return _f
    def dn_block(self, k=2):
        def _f(x):
            x = tf.keras.layers.MaxPool2D(k)(x)
            return x
        return _f
    def up_block(self, k, filters):
        def _f(x):
            x = tf.keras.layers.UpSampling2D(size=k, interpolation="bilinear")(x)
            x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            return x
        return _f

    def generate_ds_encoder(self, x, isize, nc, dense_bottleneck=False, dense_z=512, init_filters=128):
        inputs = x

        k = 1
        x = self.conv_block(init_filters * k)(x)
        b1 = x
        x = self.dn_block(2)(x)

        k = 2
        x = self.conv_block(init_filters * k)(x)
        b2 = x
        x = self.dn_block(2)(x)

        k = 4
        x = self.conv_block(init_filters * k)(x)
        b3 = x
        x = self.dn_block(2)(x)

        k = 8
        x = self.conv_block(init_filters * k)(x)
        b4 = x
        x = self.dn_block(2)(x)

        k = 8
        x = self.conv_block(init_filters * k)(x)
        outputs = x
        
        return inputs, [b1, b2, b3, b4, outputs]

    def generate_ds_decoder(self, xinputs, nc, init_filters=128):
        x = xinputs.pop()
        inputs = x

        k = 8
        k1 = 8
        x = self.up_block(2, init_filters * k)(x)
        xs = xinputs.pop()
        x = tf.keras.layers.Concatenate(axis=-1)([x, xs])
        x = self.conv_block(init_filters * k, filters1=(init_filters * k1))(x)

        k = 4
        k1 = 4
        x = self.up_block(2, init_filters * k)(x)
        xs = xinputs.pop()
        x = tf.keras.layers.Concatenate(axis=-1)([x, xs])
        x = self.conv_block(init_filters * k, filters1=(init_filters * k1))(x)

        k = 2
        k1 = 2
        x = self.up_block(2, init_filters * k)(x)
        xs = xinputs.pop()
        x = tf.keras.layers.Concatenate(axis=-1)([x, xs])
        x = self.conv_block(init_filters * k, filters1=(init_filters * k1))(x)

        k = 1
        k1 = 1
        x = self.up_block(2, init_filters * k)(x)
        xs = xinputs.pop()
        x = tf.keras.layers.Concatenate(axis=-1)([x, xs])
        x = self.conv_block(init_filters * k, filters1=(init_filters * k1))(x)

        x = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.Activation(tf.nn.sigmoid)(x)
        outputs = x

        return inputs, outputs
