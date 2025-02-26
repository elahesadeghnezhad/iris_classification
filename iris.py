import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# building the model
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(3, activation='softmax')  #! 3 output classes
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.2f}")

#classification report
print(classification_report(y_test, y_pred_classes, target_names=iris.target_names))

#new Example
new_samples = np.array([[5.1, 3.5, 1.4, 0.2],
                        [6.2, 2.9, 4.3, 1.3],
                        [7.7, 3.0, 6.1, 2.3]])
predictions = model.predict(new_samples)
predicted_classes = np.argmax(predictions, axis=1)

species_names = ['Setosa', 'Versicolor', 'Virginica']
predicted_species = [species_names[i] for i in predicted_classes]

print("Predicted species for new samples:")
for sample, species in zip(new_samples, predicted_species):
    print(f"Sample: {sample} -> Predicted species: {species}")
