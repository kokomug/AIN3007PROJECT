import pandas as pd
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, 
    EfficientNetB3, EfficientNetB4, MobileNetV2, MobileNetV3Small,
    ResNet50V2, NASNetMobile, DenseNet121, Xception
)
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Configuration (easily adjustable) ---
CONFIG = {
    # Data parameters
    'CSV_PATH': 'Data/Processed CSV\'s/train_2020_and_2019_withPateintDetail_9_labels.csv',
    'IMAGE_BASE_DIR': '512x512',
    'SUBSET_SIZE': 0.2,  # Use 20% of the data for faster training
    'NUM_CLASSES': 8,  # Updated from 9 to 8 to match the actual dataset
    
    # Training parameters
    'BATCH_SIZE': 16,
    'EPOCHS': 15,
    'PATIENCE_EARLY_STOPPING': 5,
    'PATIENCE_REDUCE_LR': 3,
    'VALIDATION_SPLIT': 0.2,
    
    # Models configuration - each model with its specific parameters
    'MODELS': {
        'efficientnet_b4': {  # Current model from codebase
            'model_fn': EfficientNetB4,
            'preprocess_fn': efficientnet_preprocess,
            'img_size': 380,  # Original size from codebase
            'weights_file': 'efficientnet_b4_subset_best.keras',
            'lr': 3e-5  # Original learning rate from codebase
        },
        'efficientnet_b0': {
            'model_fn': EfficientNetB0,
            'preprocess_fn': efficientnet_preprocess,
            'img_size': 224,
            'weights_file': 'efficientnet_b0_best.keras',
            'lr': 1e-4
        },
        'efficientnet_b1': {
            'model_fn': EfficientNetB1,
            'preprocess_fn': efficientnet_preprocess,
            'img_size': 240,
            'weights_file': 'efficientnet_b1_best.keras',
            'lr': 1e-4
        },
        'efficientnet_b2': {
            'model_fn': EfficientNetB2,
            'preprocess_fn': efficientnet_preprocess,
            'img_size': 260,
            'weights_file': 'efficientnet_b2_best.keras',
            'lr': 1e-4
        },
        'mobilenet_v2': {
            'model_fn': MobileNetV2,
            'preprocess_fn': mobilenet_preprocess,
            'img_size': 224,
            'weights_file': 'mobilenet_v2_best.keras',
            'lr': 5e-5
        },
        'mobilenet_v3_small': {
            'model_fn': MobileNetV3Small,
            'preprocess_fn': mobilenet_preprocess,
            'img_size': 224,
            'weights_file': 'mobilenet_v3_small_best.keras',
            'lr': 5e-5
        },
        'densenet121': {
            'model_fn': DenseNet121,
            'preprocess_fn': densenet_preprocess,
            'img_size': 224,
            'weights_file': 'densenet121_best.keras',
            'lr': 5e-5
        },
        'xception': {
            'model_fn': Xception,
            'preprocess_fn': xception_preprocess,
            'img_size': 299,
            'weights_file': 'xception_best.keras',
            'lr': 1e-5
        }
    }
}

# Default model to use if not specified
DEFAULT_MODEL = 'efficientnet_b4'  # Change default to current model

def create_model(model_name):
    """Create a model with the specified backbone using the project's architecture"""
    if model_name not in CONFIG['MODELS']:
        print(f"Model {model_name} not found. Using {DEFAULT_MODEL} instead.")
        model_name = DEFAULT_MODEL
    
    model_config = CONFIG['MODELS'][model_name]
    img_size = model_config['img_size']
    
    # Create base model
    base_model = model_config['model_fn'](
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Add custom top layers - matching the architecture used in original codebase (from pre_train.py)
    model = keras.Sequential()
    model.add(base_model)
    model.add(Flatten(name='top_flatten'))
    model.add(Dense(500, activation='relu', name='dense_500'))
    model.add(Dense(256, activation='relu', name='dense_256'))
    model.add(Dense(CONFIG['NUM_CLASSES'], activation='softmax', name='output_layer'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=model_config['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, img_size, model_config['preprocess_fn']

def load_and_preprocess_data(model_name, subset_size=None):
    """Load and preprocess data for the specified model"""
    if model_name not in CONFIG['MODELS']:
        print(f"Model {model_name} not found. Using {DEFAULT_MODEL} instead.")
        model_name = DEFAULT_MODEL
    
    model_config = CONFIG['MODELS'][model_name]
    img_size = model_config['img_size']
    preprocess_fn = model_config['preprocess_fn']
    
    print("=== Starting Data Loading Process ===")
    
    # Load CSV data
    if not os.path.exists(CONFIG['CSV_PATH']):
        print(f"Error: CSV file not found at {CONFIG['CSV_PATH']}")
        sys.exit(1)
    if not os.path.isdir(CONFIG['IMAGE_BASE_DIR']):
        print(f"Error: Image directory '{CONFIG['IMAGE_BASE_DIR']}' not found")
        sys.exit(1)
    
    df = pd.read_csv(CONFIG['CSV_PATH'])
    print(f"Loaded {len(df)} total images")
    
    # Create subset if specified
    if subset_size is not None and subset_size < 1.0:
        print(f"\nCreating {subset_size*100}% subset...")
        df_subset, _ = train_test_split(df, 
                                      train_size=subset_size, 
                                      stratify=df['diagnosis'],
                                      random_state=42)
        df = df_subset
        print(f"Subset size: {len(df)} images")
    
    # Add image paths
    df['image_path'] = df['image'].apply(lambda x: os.path.join(CONFIG['IMAGE_BASE_DIR'], f'{x}.jpg'))
    
    # Split into train/val
    print("\nSplitting into train/validation sets...")
    train_df, val_df = train_test_split(df,
                                      test_size=CONFIG['VALIDATION_SPLIT'],
                                      stratify=df['diagnosis'],
                                      random_state=42)
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    # Create data generators - using similar augmentation as in original codebase
    print("\nSetting up data generators...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='diagnosis',
        target_size=(img_size, img_size),
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='diagnosis',
        target_size=(img_size, img_size),
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

def train_model(model_name, subset_size=None):
    """Train the specified model"""
    if model_name not in CONFIG['MODELS']:
        print(f"Model {model_name} not found. Using {DEFAULT_MODEL} instead.")
        model_name = DEFAULT_MODEL
    
    model_config = CONFIG['MODELS'][model_name]
    weights_file = model_config['weights_file']
    
    # Create model
    model, img_size, _ = create_model(model_name)
    print(f"\nModel: {model_name}, Input size: {img_size}x{img_size}")
    
    # Load data
    train_generator, validation_generator = load_and_preprocess_data(model_name, subset_size)
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            weights_file,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['PATIENCE_EARLY_STOPPING'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=CONFIG['PATIENCE_REDUCE_LR'],
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(f"{model_name}_training_log.csv")
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=CONFIG['EPOCHS'],
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history plot
    save_training_history(history, model_name)
    
    # Evaluate model
    print("\nEvaluating model on validation data...")
    evaluate_model(model, validation_generator, model_name)
    
    print(f"\nTraining complete. Model saved to: {weights_file}")
    
    return model, history

def save_training_history(history, model_name):
    """Save training history as a plot"""
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
    plt.savefig(f"{model_name}_training_history.png")
    plt.close()

def evaluate_model(model, validation_generator, model_name):
    """Evaluate model performance with detailed metrics"""
    validation_generator.reset()
    
    # Get true labels
    y_true = []
    batch_count = 0
    for _, labels in validation_generator:
        y_true.extend(np.argmax(labels, axis=1))
        batch_count += 1
        if batch_count >= len(validation_generator):
            break
    
    # Get predictions
    validation_generator.reset()
    y_pred = model.predict(validation_generator, steps=len(validation_generator), verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Get class names
    class_indices = validation_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    labels = [class_names[i] for i in range(len(class_names))]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Print metrics
    print(f"\n===== Model Evaluation Results for {model_name} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)
    
    # Save classification report to file
    with open(f"{model_name}_classification_report.txt", "w") as f:
        f.write(f"===== Model Evaluation Results for {model_name} =====\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def unfreeze_and_finetune(model, model_name, train_generator, validation_generator):
    """Unfreeze the model and fine-tune it"""
    # Unfreeze the base model
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # Check if it's the base model
            # Keep BatchNorm layers frozen as in original code
            for inner_layer in layer.layers:
                if not isinstance(inner_layer, keras.layers.BatchNormalization):
                    inner_layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['MODELS'][model_name]['lr'] / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks for fine-tuning
    callbacks = [
        ModelCheckpoint(
            f"{model_name}_finetuned.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['PATIENCE_EARLY_STOPPING'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=CONFIG['PATIENCE_REDUCE_LR'],
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(f"{model_name}_finetuning_log.csv")
    ]
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    history_ft = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=5,  # Fewer epochs for fine-tuning
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save fine-tuning history plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_ft.history['accuracy'])
    plt.plot(history_ft.history['val_accuracy'])
    plt.title('Model accuracy (fine-tuning)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history_ft.history['loss'])
    plt.plot(history_ft.history['val_loss'])
    plt.title('Model loss (fine-tuning)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_finetuning_history.png")
    plt.close()
    
    # Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model on validation data...")
    evaluate_model(model, validation_generator, f"{model_name}_finetuned")
    
    print(f"\nFine-tuning complete. Model saved to: {model_name}_finetuned.keras")
    
    return model, history_ft

def list_available_models():
    """List all available models in the configuration"""
    print("\nAvailable lightweight models:")
    for i, model_name in enumerate(CONFIG['MODELS'].keys()):
        print(f"{i+1}. {model_name}")

def main():
    """Main function"""
    print("===== Skin Cancer Classification - Lightweight Models Training =====")
    
    # List available models
    list_available_models()
    
    # Ask which model to train
    print("\nEnter the number or name of the model to train")
    print("(or leave blank to train all models):")
    model_input = input().strip()
    
    # Ask subset size
    print("\nEnter the subset size (0.0-1.0) or leave blank for default (0.2):")
    subset_input = input().strip()
    subset_size = float(subset_input) if subset_input else CONFIG['SUBSET_SIZE']
    
    # Validate subset size
    if subset_size <= 0 or subset_size > 1:
        print("Invalid subset size. Using default 0.2")
        subset_size = CONFIG['SUBSET_SIZE']
    
    # Determine which model(s) to train
    models_to_train = []
    if not model_input:
        # Train all models
        models_to_train = list(CONFIG['MODELS'].keys())
    elif model_input.isdigit():
        # Train by number
        idx = int(model_input) - 1
        if 0 <= idx < len(CONFIG['MODELS']):
            models_to_train = [list(CONFIG['MODELS'].keys())[idx]]
        else:
            print(f"Invalid model number. Using default model {DEFAULT_MODEL}")
            models_to_train = [DEFAULT_MODEL]
    else:
        # Train by name
        if model_input in CONFIG['MODELS']:
            models_to_train = [model_input]
        else:
            print(f"Model {model_input} not found. Using default model {DEFAULT_MODEL}")
            models_to_train = [DEFAULT_MODEL]
    
    # Summary of models' performance
    model_performance = {}
    
    # Train selected models
    for model_name in models_to_train:
        print(f"\n===== Training {model_name} with {subset_size*100}% of data =====")
        try:
            model, history = train_model(model_name, subset_size)
            
            # Ask if user wants to fine-tune
            print("\nDo you want to fine-tune the model? (y/n):")
            finetune_input = input().strip().lower()
            if finetune_input == 'y':
                train_generator, validation_generator = load_and_preprocess_data(model_name, subset_size)
                finetuned_model, _ = unfreeze_and_finetune(model, model_name, train_generator, validation_generator)
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
    
    # Print overall comparison if multiple models were trained
    if len(models_to_train) > 1 and model_performance:
        print("\n===== Model Performance Comparison =====")
        for model_name, metrics in model_performance.items():
            print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    print("\n===== Training complete =====")

if __name__ == "__main__":
    main() 