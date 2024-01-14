import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_addons as tfa # for adamW
from keras.applications import VGG16
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/Users/sagartripathi/Documents/Im3ny'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_dir = '/Users/sagartripathi/Documents/Im3ny/train'
test_dir = '/Users/sagartripathi/Documents/Im3ny/test'
valid_dir = '/Users/sagartripathi/Documents/Im3ny/valid'

train_datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)

valid_generator = test_datagen.flow_from_directory(valid_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)

test_generator = test_datagen.flow_from_directory(test_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)

# Check batch size
for image_batch, labels_batch in train_generator:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

class_names = train_generator.class_indices
class_names = list(class_names.keys())
print(class_names)

class_name = 'Patches'
class_dir = os.path.join(train_dir, class_name)
sample_images = random.sample(os.listdir(class_dir), 4)

plt.figure(figsize=(12, 6))
for i, image_file in enumerate(sample_images):
    plt.subplot(1, 4, i + 1)
    image = mpimg.imread(os.path.join(class_dir, image_file))
    plt.imshow(image)
    plt.title(f'{class_name} Image {i + 1}')
    plt.axis('off')

plt.show()

# Create a VGG16 model with pre-trained weights (include_top=False to exclude the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# Create your custom classification head
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(len(class_names), activation='softmax'))

# Define the AdamW optimizer
optimizer = tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)

# Compile the model with the AdamW optimizer
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=True)
print(model.summary())

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    callbacks=[early_stopping, model_checkpoint]
)

plt.figure(figsize=(12, 8))

# Training Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Training Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))

# Print the test loss and accuracy
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Get the true labels
true_labels = test_generator.classes

# Get the predicted labels
predicted_labels = model.predict(test_generator)
predicted_labels = np.argmax(predicted_labels, axis=1)

# Create the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

images, labels = next(test_generator)

indices = np.random.choice(range(len(images)), size=9)
images = images[indices]
labels = labels[indices]

predictions = model.predict(images)

class_names = list(test_generator.class_indices.keys())

plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    image = images[i]

    if image.shape[-1] == 1:
        image = np.squeeze(image)

    plt.imshow(image)

    predicted_label = np.argmax(predictions[i])

    if predicted_label == np.argmax(labels[i]):
        color = 'blue'
        result_text = "Correct"
    else:
        color = 'red'
        result_text = "Incorrect"

    label_text = "True: " + class_names[np.argmax(labels[i])] + ", Pred: " + class_names[predicted_label] + f" ({result_text})"

    plt.xlabel(label_text, color=color)


plt.show()