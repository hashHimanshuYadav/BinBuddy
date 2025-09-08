import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- Configuration ---------------- #

# Paths
dataset_dir = 'dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
model_dir = 'models'
model_path = os.path.join(model_dir, 'waste_classifier.h5')

# Training parameters
img_height = 224
img_width = 224
batch_size = 32
epochs = 10  # You can increase this based on your dataset

# ---------------- Prepare Directories ---------------- #

# Create models folder if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# ---------------- Data Generators ---------------- #

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print(f"Detected {num_classes} classes: {train_generator.class_indices}")

# ---------------- Model Definition ---------------- #

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze base model layers

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']``
)

# ---------------- Model Training ---------------- #

print("Starting training...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ---------------- Save the Model ---------------- #

model.save(model_path)
print(f"Model saved successfully at {model_path}")