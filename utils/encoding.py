import tensorflow as tf

def positional_encoding(x, L=10):
    freq = 2.0 ** tf.range(L, dtype=tf.float32)
    freq = tf.reshape(freq, (-1, 1))
    x = tf.expand_dims(x, -2) * freq
    enc = tf.concat([tf.sin(x), tf.cos(x)], axis=-1)
    return tf.reshape(enc, (x.shape[0], -1))