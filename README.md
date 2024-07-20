# iris_classification

## first We import necessary libraries (numpy, tensorflow, sklearn) and load the Iris dataset
X contains the features (sepal length, sepal width, petal length, petal width), and y contains the target labels (species).
then We split the dataset into training and testing sets using train_test_split().
X_train, X_test, y_train, and y_test represent the training and testing data.
## them we Defining the neural network model:
We create a sequential neural network model using Keras.
The model has an input layer with 4 neurons (one for each feature), followed by two hidden layers with 25 and 15 neurons, respectively.
The output layer has 3 neurons (one for each Iris species) with softmax activation.
## Compiling the model:
We compile the model with a sparse categorical cross-entropy loss function and the Adam optimizer.
We also specify that we want to track accuracy as a metric during training.
## Training the model:
We train the model using model.fit() with the training data (X_train, y_train).
The model runs for 100 epochs (iterations) with a batch size of 32.
## Evaluating the model:
After training, we evaluate the model’s performance on the test data (X_test, y_test).
We calculate the accuracy using accuracy_score() and print it.
## Making predictions:
We create new samples (feature values) in new_samples.
The model predicts the class probabilities for these samples using model.predict().
We convert the predicted probabilities to class indices using np.argmax().
Finally, we map the class indices back to species names and print the results.
🌼
