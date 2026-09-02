A handwritten digit recognition project built from scratch using NumPy.
The neural network is trained on the MNIST dataset and can classify handwritten digits from 0 to 9. It also includes a Pygame interface that lets you draw your own digit and see the model’s prediction.



Features:

  * Neural network implemented manually with NumPy
  * MNIST training and test data
  * One hidden layer with 128 neurons
  * ReLU activation function
  * Softmax output layer
  * Backpropagation implemented manually
  * Mini-batch gradient descent
  * Test accuracy reporting
  * Interactive Pygame drawing interface
  * Prediction confidence displayed for user-drawn digits



How it works:

  Each MNIST image is originally 28 × 28 pixels.
  The image is flattened into a vector containing 784 pixel values:

    28x 28 image -> 784 inputs -> 128 hidden neutrons -> 10 outputs -> prediction

  The network uses ReLU in the hidden layer and Softmax in the output layer.
  During training, backpropagation calculates how much each weight and bias contributed to the error. Gradient descent then updates the parameters to improve the predictions.
  The weights are initialised using He initialisation, which works well with ReLU activation functions.


  
Drawing interface:

  After the network finishes training, a Pygame window opens.
  You can draw a digit using your mouse.
  The drawing is converted into a 28 × 28 grayscale image before being passed into the neural network.



Technologies:

  * Python
  * NumPy
  * PyTorch / torchvision
  * Pygame
  * Pillow
  
  PyTorch is only used through torchvision to download and access the MNIST dataset. The neural network itself is implemented using NumPy.

  

Installation:

  Clone the repository:
  git clone https://github.com/areebali06/digit-recogniser.git

  Move into the project folder:
  cd digit-recogniser

  Install the required packages:
  python -m pip install numpy torch torchvision pygame pillow

  Run:
  python -m pip install numpy torch torchvision pygame pillow

