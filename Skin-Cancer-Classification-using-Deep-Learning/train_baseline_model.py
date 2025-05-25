import pandas as pd
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
CSV_PATH = 'Data/Processed CSV\'s/train_2020_and_2019_withPateintDetail_9_labels.csv'
IMAGE_BASE_DIR = '512x512'
IMAGE_SIZE = (380, 380)
BATCH_SIZE = 8
EPOCHS = 15
NUM_CLASSES = 9
MODEL_WEIGHTS_FILE = 'efficientnetb4_baseline_20percent_best.keras'
PATIENCE_EARLY_STOPPING = 5
PATIENCE_REDUCE_LR = 3
SUBSET_SIZE = 0.2

print("=== Starting Training Process ===")
print(f"Using {SUBSET_SIZE*100}% of the dataset")

# --- Step 1: Load and Prepare Data ---
print("\n1. Loading data...")
if not os.path.exists(CSV_PATH):
    print(f"Error: CSV file not found at {CSV_PATH}")
    sys.exit(1)
if not os.path.isdir(IMAGE_BASE_DIR):
    print(f"Error: Image directory '{IMAGE_BASE_DIR}' not found")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} total images")

# Create subset
print(f"\n2. Creating {SUBSET_SIZE*100}% subset...")
df_subset, _ = train_test_split(df, 
                               train_size=SUBSET_SIZE, 
                               stratify=df['diagnosis'],
                               random_state=42)
print(f"Subset size: {len(df_subset)} images")

# Add image paths
df_subset['image_path'] = df_subset['image'].apply(lambda x: os.path.join(IMAGE_BASE_DIR, f'{x}.jpg'))

# Split into train/val
print("\n3. Splitting into train/validation sets...")
train_df, val_df = train_test_split(df_subset,
                                   test_size=0.2,
                                   stratify=df_subset['diagnosis'],
                                   random_state=42)
print(f"Training set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")

# --- Step 2: Data Generators ---
print("\n4. Setting up data generators...")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8,1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='diagnosis',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='diagnosis',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- Step 3: Model Setup ---
print("\n5. Building model...")
base_model = EfficientNetB4(
    include_top=False,
    weights='imagenet',
    input_shape=IMAGE_SIZE + (3,)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ModelCheckpoint(
        MODEL_WEIGHTS_FILE,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE_EARLY_STOPPING,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=PATIENCE_REDUCE_LR,
        min_lr=1e-6,
        verbose=1
    )
]

# --- Step 4: Training ---
print("\n6. Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(val_df) // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# --- Step 5: Save Results ---
print("\n7. Saving results...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history_20percent.png')
plt.close()

print("\n=== Training Complete ===")
print(f"Model saved to: {MODEL_WEIGHTS_FILE}")
print("Training history plot saved to: training_history_20percent.png") 