import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, ReLU #type: ignore
from sklearn.model_selection import train_test_split
import numpy as np


class NeuralNet:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.model = self._build_model(input_size, hidden_sizes, output_size)

    def _build_model(self, input_size, hidden_sizes, output_size):
        model = Sequential()
        model.add(Dense(hidden_sizes[0], input_dim=input_size))
        model.add(ReLU())
        for hidden_size in hidden_sizes[1:]:
            model.add(Dense(hidden_size))
            model.add(ReLU())
        model.add(Dense(output_size, activation='softmax'))  # For classification
        return model

    def freeze_layers(self, freeze_mask):
        for i, layer in enumerate(self.model.layers[:-1]):  # Skip the output layer
            if i < len(freeze_mask):
                layer.trainable = not freeze_mask[i]
        # Recompile the model after modifying trainable attributes
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def train_model(self, train_data, val_data, epochs=10, learning_rate=0.001):
        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Train the model
        history = self.model.fit(
            x=train_data[0],
            y=train_data[1],
            validation_data=val_data,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )

        return history.history
    
    def reinitialize_weights(self):
            """
            Reinitialize the weights of the TensorFlow/Keras model.
            """
            for layer in self.model.layers:
                if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                    layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
                    layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))
                    
    # Generate Mock Data
    def generate_mock_data(input_size, num_classes, num_samples=1000, validation_split=0.2):
        X = np.random.rand(num_samples, input_size).astype(np.float32)
        y = np.random.randint(0, num_classes, size=num_samples)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

        return (X_train, y_train), (X_val, y_val)


# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 10
    hidden_sizes = [32, 16, 8]
    output_size = 3
    epochs = 15
    learning_rate = 0.001

    # Initialize the network
    net = NeuralNet(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    net.summary()

    # Generate mock data
    (X_train, y_train), (X_val, y_val) = net.generate_mock_data(input_size, output_size)

    # Train the network
    history = net.train_model((X_train, y_train), (X_val, y_val), epochs=epochs, learning_rate=learning_rate)
