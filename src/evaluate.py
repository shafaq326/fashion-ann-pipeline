import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

model = tf.keras.models.load_model("models/model.h5")

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)

metrics = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Fashion-MNIST Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()

print("Evaluation completed successfully.")
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
print("Metrics saved to metrics.json")
print("Confusion matrix saved to confusion_matrix.png")
