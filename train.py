import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from LeNet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="alamat lokasi dataset")
ap.add_argument("-m", "--model", required=True, help="alamat lokasi model akan disimpan")
ap.add_argument("-p", "--plot", type=str, default="plot.png",help="nama plot yang akan dihasilkan")
args = vars(ap.parse_args())

EPOCHS = 25
INIT_LR = 2e-4
BS = 32

print("[INFO] loading images . . .")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(30, 30))
	image = img_to_array(image)
	data.append(image)
	
	label = imagePath.split(os.path.sep)[-2]
	if label == "positif":
		label = 1
	else:
		label = 0

	labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.4, random_state=42)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model . . .")
model = LeNet.build(width=30, height=30, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network . . .")
# H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX,testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX,testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

print("[INFO] serializing network . . .")
model.save(args["model"], save_format="h5")

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])