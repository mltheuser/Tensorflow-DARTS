import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from tensorflow.keras import datasets, layers

from src.mixed_ops import ArchitectureSearchModel, BinaryMixedOp, ContinuousMixedOp, BinaryMaskedMixedOp, \
    BinaryMovingAverageMixedOp, BinaryProxylessNASStyleMixedOp, MixedOp

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

MixedOpUsed = BinaryMaskedMixedOp


class CustomSequential(tf.keras.layers.Layer):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list

    def __call__(self, x, training, *args, **kwargs):
        for layer in self.layer_list:
            x = layer(x, training=training)
        return x


def convolution_options(): return [
    CustomSequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dense(64)
    ]),
    CustomSequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Dense(64)
    ]),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
]


def pooling_options(): return [
    layers.MaxPooling2D((2, 2)),
    layers.AveragePooling2D((2, 2)),
]


def dense_options(): return [
    tf.keras.layers.Dense(64, activation=None),
    tf.keras.layers.Dense(64, activation='relu'),
    CustomSequential([
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(64)
    ]),
    CustomSequential([
        tf.keras.layers.Dense(2000, activation='relu'),
        tf.keras.layers.Dense(2000, activation='relu'),
        tf.keras.layers.Dense(64)
    ]),
]


class CustomModel(ArchitectureSearchModel):
    def __init__(self):
        super().__init__()

        self.stochastic_train = True

        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.memory_footprint_tracker = tf.keras.metrics.Mean(name="memory_footprint")

        self.internal_layers = [
            MixedOpUsed(
                convolution_options(),
                num_on_samples=1, num_eval_samples=1,
                name='conv_mixed_op_1',
            ),
            MixedOpUsed(
                pooling_options(),
                num_on_samples=1, num_eval_samples=1,
                name='pool_mixed_op_1',
            ),
            MixedOpUsed(
                convolution_options(),
                num_on_samples=1, num_eval_samples=1,
                name='conv_mixed_op_2',
            ),
            MixedOpUsed(
                pooling_options(),
                num_on_samples=1, num_eval_samples=1,
                name='pool_mixed_op_2',
            ),
            MixedOpUsed(
                convolution_options(),
                num_on_samples=1, num_eval_samples=1,
                name='conv_mixed_op_3',
            ),

            layers.Flatten(),
            MixedOpUsed(
                dense_options(),
                num_on_samples=1, num_eval_samples=1,
                name='dense_mixed_op_1',
            ),
            MixedOpUsed(
                dense_options(),
                num_on_samples=1, num_eval_samples=1,
                name='dense_mixed_op_2',
            ),
            layers.Dense(10),
        ]

    def __call__(self, x, stochastic, training, *args, **kwargs):
        for layer in self.internal_layers:
            if isinstance(layer, MixedOp):
                x = layer(x, stochastic=stochastic, training=training)
            else:
                x = layer(x, training=training)
        return x

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, stochastic=self.stochastic_train, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.accuracy_tracker.update_state(y, y_pred)
        if tf.config.list_physical_devices('GPU'):
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            self.memory_footprint_tracker.update_state(memory_info['peak'])

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, stochastic=False, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Compute our own metrics
        self.accuracy_tracker.update_state(y, y_pred)
        if tf.config.list_physical_devices('GPU'):
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            self.memory_footprint_tracker.update_state(memory_info['peak'])
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.accuracy_tracker, self.memory_footprint_tracker]


def main():
    model = CustomModel()

    model.stochastic_train = True
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        run_eagerly=True,
    )

    print(model.architecture_summary())

    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))

    test_acc, _ = model.evaluate(test_images, test_labels, verbose=2)

    print(model.architecture_summary())

    print(test_acc)

    model.stochastic_train = False
    model.train_function = None

    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))

    test_acc, _ = model.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)


if __name__ == "__main__":
    main()
