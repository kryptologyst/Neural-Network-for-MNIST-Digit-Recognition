# Neural Network for MNIST Digit Recognition

This project demonstrates how to build, train, and evaluate a neural network for classifying handwritten digits from the MNIST dataset. It also includes a simple graphical user interface (GUI) for real-time digit recognition.

## Features

- **Data Loading and Preprocessing:** Loads and preprocesses the MNIST dataset for training and testing.
- **Model Building:** Constructs a sequential neural network model using TensorFlow/Keras.
- **Model Training:** Trains the model on the MNIST training data and saves it for later use.
- **Model Evaluation:** Evaluates the trained model's accuracy on the test dataset.
- **GUI for Prediction:** A user-friendly interface where you can draw a digit and get a real-time prediction from the trained model.

## Technologies Used

- **Python 3:** The core programming language used for this project.
- **TensorFlow/Keras:** For building and training the neural network.
- **NumPy:** For numerical operations and data manipulation.
- **Pillow:** For image processing in the GUI.
- **Tkinter:** For creating the graphical user interface.

## Setup and Installation

To run this project, you'll need to have Python 3 installed on your system. You can then follow these steps to set up the environment and run the application:

1.  **Clone the repository (or download the files):**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Train the model:**

    Before you can use the GUI for predictions, you need to train the model. Run the following command in your terminal:

    ```bash
    python3 main.py
    ```

    This will train the model and save it as `mnist_model.h5` in the project directory.

2.  **Launch the GUI:**

    Once the model is trained, you can start the digit recognizer application:

    ```bash
    python3 gui.py
    ```

    This will open a window where you can draw a digit on the canvas. Click the "Predict" button to see the model's prediction, or "Clear" to reset the canvas.
# Neural-Network-for-MNIST-Digit-Recognition
