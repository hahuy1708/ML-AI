# CNN Exercise
## Implementing and Evaluating LeNet-5 on MNIST and CIFAR-10

The goal of this exercise is to build, train, and evaluate a Convolutional Neural Network (CNN) based on the classic LeNet-5 architecture. You will apply this model to two distinct datasets: the MNIST database of handwritten digits and the CIFAR-10 dataset of natural images. This exercise has three tasks.

The assignment submission link will be provided later. You must submit a link to your completed Jupyter Notebook (`.ipynb`). Ensure your notebook has been fully executed with all cell outputs, markdown explanations, and requested plots visible. Save the file in your personal Google Drive and set the sharing permissions to **Public (Anyone with the link can view)** before submitting.

---

## Task 1: Data Acquisition and Exploration

You must perform the following data preparation and visualization steps for both the MNIST and CIFAR-10 datasets using `tensorflow.keras.datasets`:

1. **Load the Data:** Import the train and test splits for both datasets.
2. **Data Summarization:** Print out the exact number of samples in the training and testing sets. Print the shape of a single image and list the names of the 10 label classes.
3. **Data Visualization:** Create a plot displaying a grid of images. For each of the 10 classes, randomly select and plot 5 sample images along with their corresponding class labels.
4. **Normalization:** Scale the pixel values of all images so they fall within the range of `[0, 1]`.
5. **Label Encoding:** Apply the one-hot encoding representation to the target labels.
6. **Validation Split:** Further partition your original training data into a new training set and a validation set using an **80/20** ratio.

---

## Task 2: Building Model Architecture

Construct a Convolutional Neural Network inspired by the LeNet-5 architecture. The core architecture should generally follow this structure:

| Layer Type | Features / Filters | Activation |
|---|---|---|
| Convolutional (Conv2D) | 6 filters, 5×5 kernel, stride 1 | Tanh |
| Average Pooling | 2×2 pool size, stride 2 | - |
| Convolutional (Conv2D) | 16 filters, 5×5 kernel, stride 1 | Tanh |
| Average Pooling | 2×2 pool size, stride 2 | - |
| Fully Connected (Dense) | 120 units | Tanh |
| Fully Connected (Dense) | 84 units | Tanh |
| Output (Dense) | 10 units (Classes) | Softmax |

Because MNIST (28×28 grayscale) and CIFAR-10 (32×32 RGB) have different input dimensions and channels, you must adapt the `input_shape` of your first layer accordingly.

---

## Task 3: Training and Evaluation

Execute the training and evaluation pipeline on both datasets independently:

1. **Training:** Train the model for **10–15 epochs**.
2. **Epoch Logging:** After each epoch, output the **Loss, Precision, and Recall** values for both the Training set and the Validation set.
3. **Testing:** Once the final epoch is complete, evaluate the fully trained model on the unseen Test set.
4. **Visualizing Metrics:** Plot three separate line graphs showing:
   - Training Loss vs. Validation Loss across epochs.
   - Training Precision vs. Validation Precision across epochs.
   - Training Recall vs. Validation Recall across epochs.
5. **Confusion Matrix:** Generate and plot a Confusion Matrix based on the test set predictions to visualize misclassifications.
6. **Sample Predictions:** Query your test set results to find specific examples. Plot **5 correctly classified** images and **5 incorrectly classified** images. For each image, clearly print the input image, the model's predicted label, and the actual true ground-truth label.