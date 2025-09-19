import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    """Loads and preprocesses the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    return (x_train, y_train), (x_test, y_test)

def build_model():
    """Builds the neural network model."""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train):
    """Trains the model."""
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)
    model.save('mnist_model.h5')

def evaluate_model(model, x_test, y_test):
    """Evaluates the model and prints the test accuracy."""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest Accuracy: {test_acc:.4f}')


def main():
    """Main function to run the MNIST digit recognition task."""
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    try:
        model = load_model('mnist_model.h5')
        print('Loaded model from disk.')
    except (IOError, ImportError):
        print('No pre-trained model found. Training a new one.')
        model = build_model()
        train_model(model, x_train, y_train)

    evaluate_model(model, x_test, y_test)

if __name__ == '__main__':
    main()