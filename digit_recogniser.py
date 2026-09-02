import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import pygame
from PIL import Image


# load the MNIST data
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True
)

# convert the pytorch data into numpy arrays
X_train = train_data.data.numpy()
y_train = train_data.targets.numpy()

X_test = test_data.data.numpy()
y_test = test_data.targets.numpy()

# flatten each 28x28 image into 784 values
X_train = X_train.reshape(-1, 784).astype("float32")
X_test = X_test.reshape(-1, 784).astype("float32")

# pixel values are originally between 0 and 255
X_train = X_train / 255
X_test = X_test / 255


def initialise():
    # 784 inputs -> 128 hidden neurons -> 10 outputs
    W1 = np.random.randn(784, 128) * np.sqrt(2 / 784)
    b1 = np.zeros((1, 128))

    W2 = np.random.randn(128, 10) * np.sqrt(2 / 128)
    b2 = np.zeros((1, 10))

    return W1, b1, W2, b2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return z > 0


def softmax(z):
    # subtract the largest value so exp() doesn't get too large
    z = z - np.max(z, axis=1, keepdims=True)

    exp_values = np.exp(z)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


def forward_prop(W1, b1, W2, b2, X):
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2


def back_prop(X, y, z1, a1, a2, W2):
    samples = X.shape[0]

    dz2 = a2.copy()

    # reduce the probability of the correct class by 1
    dz2[np.arange(samples), y] -= 1
    dz2 /= samples

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


def update_parameters(
    W1,
    b1,
    W2,
    b2,
    dW1,
    db1,
    dW2,
    db2,
    learning_rate
):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2


def get_predictions(output):
    return np.argmax(output, axis=1)


def get_accuracy(predictions, answers):
    return np.mean(predictions == answers)


def train(X, y, epochs=10, learning_rate=0.1, batch_size=128):
    W1, b1, W2, b2 = initialise()

    number_of_samples = X.shape[0]

    for epoch in range(epochs):

        # shuffle the training data each time
        indices = np.random.permutation(number_of_samples)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, number_of_samples, batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            z1, a1, z2, a2 = forward_prop(
                W1,
                b1,
                W2,
                b2,
                X_batch
            )

            dW1, db1, dW2, db2 = back_prop(
                X_batch,
                y_batch,
                z1,
                a1,
                a2,
                W2
            )

            W1, b1, W2, b2 = update_parameters(
                W1,
                b1,
                W2,
                b2,
                dW1,
                db1,
                dW2,
                db2,
                learning_rate
            )

        # check accuracy after every epoch
        _, _, _, output = forward_prop(
            W1,
            b1,
            W2,
            b2,
            X
        )

        predictions = get_predictions(output)
        accuracy = get_accuracy(predictions, y)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    return W1, b1, W2, b2


# train the network
W1, b1, W2, b2 = train(
    X_train,
    y_train,
    epochs=10,
    learning_rate=0.1
)


# test it on images it hasn't trained on
_, _, _, test_output = forward_prop(
    W1,
    b1,
    W2,
    b2,
    X_test
)

test_predictions = get_predictions(test_output)
test_accuracy = get_accuracy(test_predictions, y_test)

print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")


# drawing window

pygame.init()

window_width = 500
window_height = 420
canvas_size = 280

screen = pygame.display.set_mode(
    (window_width, window_height)
)

pygame.display.set_caption("Digit Recogniser")

font = pygame.font.SysFont("Arial", 28)
small_font = pygame.font.SysFont("Arial", 20)

canvas_x = (window_width - canvas_size) // 2
canvas_y = 20

canvas_rect = pygame.Rect(
    canvas_x,
    canvas_y,
    canvas_size,
    canvas_size
)

drawing_surface = pygame.Surface(
    (canvas_size, canvas_size)
)

drawing_surface.fill((0, 0, 0))

prediction_text = "Draw a digit"

drawing = False
brush_size = 18


def clear_canvas():
    drawing_surface.fill((0, 0, 0))


def prepare_image():

    # turn the pygame drawing into an image
    raw_data = pygame.image.tostring(
        drawing_surface,
        "RGB"
    )

    image = Image.frombytes(
        "RGB",
        (canvas_size, canvas_size),
        raw_data
    )

    image = image.convert("L")

    # find the area that actually contains the digit
    box = image.getbbox()

    if box is None:
        return None

    digit = image.crop(box)

    # MNIST digits are roughly 20x20 inside a 28x28 image
    digit.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    processed = Image.new(
        "L",
        (28, 28),
        0
    )

    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    processed.paste(digit, (x, y))

    image_array = np.array(
        processed,
        dtype=np.float32
    )

    image_array = image_array / 255

    # network expects 784 inputs
    image_array = image_array.reshape(1, 784)

    return image_array


def predict_digit():

    image = prepare_image()

    if image is None:
        return None, None

    _, _, _, output = forward_prop(
        W1,
        b1,
        W2,
        b2,
        image
    )

    prediction = get_predictions(output)[0]

    confidence = output[0, prediction]

    return prediction, confidence


running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:

                mouse_x, mouse_y = event.pos

                if canvas_rect.collidepoint(
                    mouse_x,
                    mouse_y
                ):
                    drawing = True

        elif event.type == pygame.MOUSEBUTTONUP:

            if event.button == 1:
                drawing = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_c:

                clear_canvas()
                prediction_text = "Draw a digit"

            elif event.key == pygame.K_SPACE:

                prediction, confidence = predict_digit()

                if prediction is None:
                    prediction_text = "Draw something first"

                else:
                    prediction_text = (
                        f"Prediction: {prediction} "
                        f"({confidence * 100:.1f}%)"
                    )

    if drawing:

        mouse_x, mouse_y = pygame.mouse.get_pos()

        x = mouse_x - canvas_x
        y = mouse_y - canvas_y

        if 0 <= x < canvas_size and 0 <= y < canvas_size:

            pygame.draw.circle(
                drawing_surface,
                (255, 255, 255),
                (x, y),
                brush_size
            )

    screen.fill((35, 35, 35))

    screen.blit(
        drawing_surface,
        (canvas_x, canvas_y)
    )

    result_text = font.render(
        prediction_text,
        True,
        (255, 255, 255)
    )

    result_rect = result_text.get_rect(
        center=(window_width // 2, 335)
    )

    screen.blit(
        result_text,
        result_rect
    )

    instructions = small_font.render(
        "SPACE = Predict    C = Clear",
        True,
        (200, 200, 200)
    )

    instructions_rect = instructions.get_rect(
        center=(window_width // 2, 385)
    )

    screen.blit(
        instructions,
        instructions_rect
    )

    pygame.display.flip()


pygame.quit()