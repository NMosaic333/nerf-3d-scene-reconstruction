import tensorflow as tf
class NeRF(tf.keras.Model):
    def __init__(self, depth=8, width=256, skip_layer=5):
        super().__init__()

        self.depth = depth
        self.width = width

        self.layers_pos = [
            tf.keras.layers.Dense(width, activation='relu')
            for _ in range(depth)
        ]

        self.skip_layer = skip_layer

        self.sigma_layer = tf.keras.layers.Dense(1, activation='relu')
        self.feature_layer = tf.keras.layers.Dense(256)

        self.color_layer1 = tf.keras.layers.Dense(128, activation='relu')
        self.color_layer2 = tf.keras.layers.Dense(3, activation='sigmoid')

    def call(self, x, d):
        h = x
        inputs = x

        for i in range(self.depth):
            h = self.layers_pos[i](h)
            if i == self.skip_layer:
                h = tf.concat([h, inputs], -1)

        sigma = self.sigma_layer(h)
        feat = self.feature_layer(h)

        h_color = tf.concat([feat, d], -1)
        h_color = self.color_layer1(h_color)
        rgb = self.color_layer2(h_color)

        return tf.concat([rgb, sigma], axis=-1)  # (B*N, 4)