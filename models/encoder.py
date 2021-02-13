import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer


class ConvLayer(Layer):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = tf.keras.layers.Conv1D(
                                  filters=c_in,
                                  kernel_size=3,
                                  padding='causal')
        self.norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ELU()
        self.maxPool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2)

    def call(self, x, **kargs):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x


class EncoderLayer(Layer):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1)
        self.conv2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.activation = tf.keras.activations.relu if activation == "relu" else tf.keras.activations.gelu

    def call(self, x, attn_mask=None):
        # x [B, L, D]
        x = x + self.dropout(self.attention(
            [x, x, x],
            attn_mask = attn_mask
        ))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        return self.norm2(x+y)


class Encoder(Layer):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers if conv_layers is not None else None
        self.norm = norm_layer

    def call(self, x, attn_mask=None):
        # x [B, L, D]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

