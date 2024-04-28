import numpy as np
import os
import argparse
import pickle
from utils.data import get_sequence_data
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Reshape, Dropout, TimeDistributed, Flatten, Input
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Concatenate

# Configurations
DATASET_PATH = "data"
OUTPUT_PATH = "model"
FRAMES_PER_SEQ = 30
LABELS = set(["collision", "safe", "tailgating", "weaving"])

NUM_EPOCHS = 25 #150
BATCH_SIZE = 21 #2048
PATIENCE = 10 #50

# Load Input Data
print("[INFO] loading sequence data...")
data, labels = get_sequence_data(DATASET_PATH, LABELS, FRAMES_PER_SEQ)

training_data = np.array(data["training"])
training_labels = np.array(labels["training"])
validation_data = np.array(data["validation"])
validation_labels = np.array(labels["validation"])

print(f"[INFO] number of sequences in training_data: {len(training_data)}")
print(f"[INFO] number of sequences in validation_data: {len(validation_data)}")

lb = LabelBinarizer()
training_labels = lb.fit_transform(training_labels)
validation_labels = lb.transform(validation_labels)

# Chunk and Reshape Data
trainX = training_data
testX = validation_data
trainY = training_labels
testY = validation_labels

trainX = trainX.reshape(trainX.shape[0], FRAMES_PER_SEQ, -1)
testX = testX.reshape(testX.shape[0], FRAMES_PER_SEQ, -1)

# Load ResNet50
resnet = ResNet50(weights="imagenet", include_top=False, pooling='avg')

# Freeze layers
for layer in resnet.layers:
    layer.trainable = False

# Define LSTM model with ResNet50 processing
inputs = Input(shape=(FRAMES_PER_SEQ, 224, 224, 3))
frames = []
for i in range(FRAMES_PER_SEQ):
    frame = inputs[:, i, :, :, :]
    frame = resnet(frame)
    frames.append(frame)


# Calculate num_features
# num_features = resnet.output_shape[1]

# Define the input layer for the LSTM
lstm_input = TimeDistributed(Flatten())(frames)  # Flatten the output of ResNet50 and apply TimeDistributed

# Define the LSTM layer
lstm_output = LSTM(units=64)(lstm_input)  # You can adjust the number of LSTM units as needed
lstm_output = Dropout(0.5)(lstm_output)
outputs = Dense(len(lb.classes_), activation="softmax")(lstm_output)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
print("[INFO] compiling model...")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
print("[INFO] training model...")
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[
    EarlyStopping(monitor='val_loss', patience=PATIENCE),
    ModelCheckpoint(os.path.join(OUTPUT_PATH, 'checkpoint.keras'), save_best_only=True, save_weights_only=False)
])
NUM_EPOCHS = len(history.history['loss'])

# Evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, NUM_EPOCHS), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, NUM_EPOCHS), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, NUM_EPOCHS), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, NUM_EPOCHS), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(OUTPUT_PATH, "plot.png"))

# Save Output to Disk
print("[INFO] serializing model...")
model.save(os.path.join(OUTPUT_PATH, "model.keras"))

with open(os.path.join(OUTPUT_PATH, "labels.pickle"), "wb") as f:
    f.write(pickle.dumps(lb))
