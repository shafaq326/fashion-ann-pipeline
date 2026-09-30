import os
import csv
import yaml
import numpy as np
import tensorflow as tf

os.makedirs("models", exist_ok=True)

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

train_params = params["train"]

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(train_params["dense_units"], activation="relu"),
    tf.keras.layers.Dropout(train_params["dropout_rate"]),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=train_params["learning_rate"]
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=train_params["epochs"],
    batch_size=train_params["batch_size"]
)

model.save("models/model.h5")

with open("models/history.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])

    for i in range(len(history.history["loss"])):
        writer.writerow([
            i + 1,
            history.history["loss"][i],
            history.history["accuracy"][i],
            history.history["val_loss"][i],
            history.history["val_accuracy"][i]
        ])

print("Training completed successfully.")
print("Model saved to models/model.h5")
print("History saved to models/history.csv")
