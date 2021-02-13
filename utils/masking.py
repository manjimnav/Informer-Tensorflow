import tensorflow as tf


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]

        mask_a = tf.linalg.band_part(tf.ones(mask_shape), 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(tf.ones(mask_shape), 0, 0)  # Diagonal matrix of 0s and 1s
        mask = tf.cast(mask_a - mask_b, dtype=tf.float32)

        self._mask = mask
        tf.stop_gradient(self._mask)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores):
        _mask = tf.ones((L, scores.shape[-1]))

        mask_a = tf.linalg.band_part(_mask, 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(_mask, 0, 0)  # Diagonal matrix of 0s and 1s
        _mask = tf.cast(mask_a - mask_b, dtype=tf.float32)

        _mask_ex = tf.broadcast_to(_mask, [B, H, L, scores.shape[-1]])
        indicator = _mask_ex[tf.range(B)[:, None, None],
                    tf.range(H)[None, :, None],
                    index, :]
        self._mask = indicator.reshape(scores.shape)

    @property
    def mask(self):
        return self._mask

if __name__ == "__main__":
    print(tf.ones((5,5)).triu(1))