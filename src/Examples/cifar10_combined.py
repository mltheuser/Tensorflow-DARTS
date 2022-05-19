import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from tensorflow.keras import datasets, layers

from src.mixed_ops import ArchitectureSearchModel, BinaryMixedOp, ContinuousMixedOp, BinaryMaskedMixedOp, \
    BinaryMovingAverageMixedOp, BinaryProxylessNASStyleMixedOp

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


MixedOpUsed = BinaryProxylessNASStyleMixedOp


class CustomModel(ArchitectureSearchModel):
    def __init__(self):
        super().__init__()

        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.memory_footprint_tracker = tf.keras.metrics.Mean(name="memory_footprint")

        self.internal_layers = [
            MixedOpUsed(
                [
                    tf.keras.layers.Dense(32, activation=None),
                    tf.keras.layers.Dense(32, activation='relu'),
                    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                ],
                num_on_samples=1, num_eval_samples=2
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),

            layers.Flatten(),
            MixedOpUsed(
                [
                    layers.Dense(64, activation='relu'),
                    layers.Dense(64, activation=None),
                ],
                num_on_samples=1, num_eval_samples=2
            ),
            layers.Dense(10),
        ]

    def __call__(self, x, training, *args, **kwargs):
        for layer in self.internal_layers:
            x = layer(x, training=training)
        return x

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
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
        y_pred = self(x, training=False)
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


model = CustomModel()

model.trainable_architecture = True

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    run_eagerly=False,
)

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels))

test_acc, _ = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)

model.trainable_architecture = False

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels))

test_acc, _ = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)
