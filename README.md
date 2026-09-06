# Handwritten Digit Recogniser

A handwritten digit recognition application built in Python using a neural network trained on the MNIST dataset. The application allows users to draw a digit and have the model predict which number from **0–9** was drawn.

The neural network is implemented using NumPy, while `torchvision` is used to download and load the MNIST dataset. A Pygame interface allows users to test the trained model with their own handwritten digits.

## Features

- Recognises handwritten digits from **0 to 9**
- Neural network implemented from scratch using **NumPy**
- Trained using the **MNIST handwritten digit dataset**
- Interactive drawing interface built with **Pygame**
- Allows users to draw their own digits for recognition
- Processes drawings into the format expected by the neural network
- Displays the model's predicted digit

## Technologies

- **Python**
- **NumPy** — neural network calculations and matrix operations
- **PyTorch / Torchvision** — loading the MNIST dataset
- **Pygame** — interactive drawing interface
- **Matplotlib** — image and training visualisation
- **Pillow** — image processing

## Neural Network

The model uses a simple feed-forward neural network:

```text
Input Layer        Hidden Layer        Output Layer
784 neurons   →    128 neurons    →    10 neurons
   (28×28)                               (0–9)
```

Each MNIST image is **28 × 28 pixels**. The image is flattened into an array containing **784 pixel values** before being passed into the neural network.

The hidden layer contains **128 neurons**, while the output layer contains **10 neurons**, corresponding to the digits 0–9.

## Dataset

The project uses the **MNIST dataset**, which contains thousands of labelled images of handwritten digits.

`torchvision` is used to download and load the dataset.

The images are converted into numerical arrays and normalised before being passed into the neural network.

## Getting Started

### Prerequisites

Python is required to run the project.

**Python 3.12 is recommended.**

Check your installed Python version with:

```bash
python3 --version
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/areebali06/digit-recogniser.git
cd digit-recogniser
```

### 2. Create a virtual environment

Using a virtual environment keeps the project's dependencies separate from other Python installations.

#### macOS / Linux

```bash
python3.12 -m venv neuralnet-env
source neuralnet-env/bin/activate
```

#### Windows

```bash
python -m venv neuralnet-env
neuralnet-env\Scripts\activate
```

### 3. Install dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- NumPy
- Matplotlib
- PyTorch
- Torchvision
- Pygame
- Pillow

## Running the Project

Make sure the virtual environment is activated.

### macOS / Linux

```bash
source neuralnet-env/bin/activate
python main.py
```

### Windows

```bash
neuralnet-env\Scripts\activate
python main.py
```

The application should then open the digit drawing interface.

## Using the Digit Recogniser

1. Draw a digit from **0–9** inside the drawing area.
2. Submit the drawing for recognition.
3. The drawing is resized and processed into the same format as the MNIST images.
4. The processed image is passed through the neural network.
5. The predicted digit is displayed.
6. Clear the canvas to draw another digit.

## How It Works

### 1. Loading the Dataset

The MNIST dataset is loaded using `torchvision`. Each image contains a handwritten digit with a corresponding label between 0 and 9.

### 2. Preprocessing

Each MNIST image has a resolution of **28 × 28 pixels**.

The image is flattened before entering the neural network:

```text
28 × 28 = 784 input values
```

Pixel values are also normalised before being processed by the model.

### 3. Neural Network

The flattened image is passed through the neural network:

```text
784 inputs → 128 hidden neurons → 10 outputs
```

The hidden layer learns patterns within the images that help distinguish between different handwritten digits.

### 4. Prediction

The output layer produces values corresponding to each possible digit.

The digit with the highest output value is selected as the model's prediction.

### 5. User-Drawn Digits

The Pygame interface allows users to draw their own digits.

The drawing is converted into a **28 × 28 representation** before being passed through the neural network for classification.

## Project Structure

```text
digit-recogniser/
├── main.py
├── requirements.txt
├── README.md
└── neuralnet-env/
```

The virtual environment should **not** be committed to the repository.

A `.gitignore` file can include:

```gitignore
neuralnet-env/
__pycache__/
*.pyc
.DS_Store
```

## What I Learned

This project helped develop my understanding of:

- Neural network fundamentals
- Forward propagation
- Training and optimisation
- Matrix operations using NumPy
- Image preprocessing
- Working with machine-learning datasets
- Python dependency management
- Python virtual environments
- Building interactive interfaces with Pygame

Implementing the neural network logic with NumPy also helped provide a better understanding of the calculations that take place behind higher-level machine-learning frameworks.

## Future Improvements

- Display confidence scores for predictions
- Improve recognition of user-drawn digits
- Visualise output probabilities
- Add training accuracy and loss graphs
- Experiment with different neural network architectures
- Compare the NumPy implementation with a PyTorch model
- Improve the drawing interface
- Save and load trained model weights

GitHub: `areebali06`
