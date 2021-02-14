import tensorflow as tf
from embed import DataEmbedding
from attn import ProbAttention, FullAttention, AttentionLayer
from encoder import ConvLayer, Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer


class Informer(tf.keras.Model):

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, batch_size,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', data='ETTh', activation='gelu'):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.seq_len = seq_len
        self.label_len = label_len
        self.batch_size = batch_size

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, data, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, data, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ],
            norm_layer=tf.keras.layers.LayerNormalization()
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=tf.keras.layers.LayerNormalization()
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = tf.keras.layers.Dense(c_out)

    def call(self, inputs, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_enc, x_dec, x_mark_enc, x_mark_dec = inputs

        x_enc.set_shape((self.batch_size, self.seq_len, x_enc.shape[2]))
        x_mark_enc.set_shape((self.batch_size, self.seq_len, x_mark_enc.shape[2]))

        x_dec.set_shape((self.batch_size, self.label_len+self.pred_len, x_dec.shape[2]))
        x_mark_dec.set_shape((self.batch_size, self.label_len+self.pred_len, x_mark_dec.shape[2]))

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    model = Informer(7, 7, 7, 96, 48, 24, 32)
    x_enc = tf.zeros((32, 96, 7))
    x_dec = tf.zeros((32, 72, 7))
    x_mark_enc = tf.zeros((32, 96, 4))
    x_mark_dec = tf.zeros((32, 72, 4))
    print(model([x_enc, x_dec, x_mark_enc, x_mark_dec]).shape)
