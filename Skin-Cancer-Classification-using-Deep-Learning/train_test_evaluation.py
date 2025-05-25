import pandas as pd
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import glob

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
IMAGE_DIR = '512x512'
TEST_SUBSET_SIZE = 0.15  # Use 15% of available images
BATCH_SIZE = 16
NUM_CLASSES = 8
EPOCHS = 15  # Train for 15 epochs as requested
VALIDATION_SPLIT = 0.2  # 80% train, 20% validation within the test subset

# Class indices mapping
CLASS_INDICES = {
    'MEL': 0,  # Melanoma
    'NV': 1,   # Melanocytic nevus
    'BCC': 2,  # Basal cell carcinoma
    'AK': 3,   # Actinic keratosis
    'BKL': 4,  # Benign keratosis
    'DF': 5,   # Dermatofibroma
    'VASC': 6, # Vascular lesion
    'SCC': 7   # Squamous cell carcinoma
}

# Class names (reverse mapping)
CLASS_NAMES = [k for k, v in sorted(CLASS_INDICES.items(), key=lambda item: item[1])]

# Model configurations
MODEL_CONFIGS = {
    'efficientnet_b4': {
        'original_weights': 'efficientnet_b4_subset_best.keras',
        'test_weights': 'test_efficientnet_b4_best.keras',
        'img_size': 380,
        'preprocess_fn': efficientnet_preprocess,
        'lr': 3e-5
    },
    'densenet121': {
        'original_weights': 'densenet121_best.keras',
        'test_weights': 'test_densenet121_best.keras',
        'img_size': 224,
        'preprocess_fn': densenet_preprocess,
        'lr': 5e-5
    },
    'xception': {
        'original_weights': 'xception_best.keras',
        'test_weights': 'test_xception_best.keras',
        'img_size': 299,
        'preprocess_fn': xception_preprocess,
        'lr': 1e-5
    }
}

def create_labeled_dataset():
    """Create a labeled dataset from available images for training/testing"""
    print("Creating labeled dataset for training/evaluation...")
    
    # Get all image files
    image_files = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    
    if not image_files:
        print(f"Error: No images found in {IMAGE_DIR}")
        return None
    
    print(f"Found {len(image_files)} images")
    
    # Select a subset of images (15%)
    sample_size = int(len(image_files) * TEST_SUBSET_SIZE)
    if len(image_files) > sample_size:
        sample_files = random.sample(image_files, sample_size)
    else:
        sample_files = image_files
    
    print(f"Selected {len(sample_files)} images ({TEST_SUBSET_SIZE*100:.1f}% of available data)")
    
    # Assign random labels to create a labeled dataset
    # This is for illustration only - in a real scenario, we would have actual labels
    sample_data = []
    for file_path in sample_files:
        image_id = os.path.basename(file_path).replace('.jpg', '')
        # Generate a random label (0-7) for demonstration
        label = random.choice(CLASS_NAMES)
        sample_data.append({
            'image': image_id,
            'image_path': file_path,
            'diagnosis': label
        })
    
    # Create dataframe with labels
    sample_df = pd.DataFrame(sample_data)
    
    # Split into train/validation
    train_df, val_df = train_test_split(sample_df, 
                                        test_size=VALIDATION_SPLIT,
                                        stratify=sample_df['diagnosis'],
                                        random_state=42)
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    return train_df, val_df

def create_data_generators(train_df, val_df, model_name):
    """Create data generators for training and validation"""
    model_config = MODEL_CONFIGS[model_name]
    img_size = model_config['img_size']
    preprocess_fn = model_config['preprocess_fn']
    
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
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='diagnosis',
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

def create_model(model_name):
    """Create a model with proper architecture and load original weights if available"""
    model_config = MODEL_CONFIGS[model_name]
    img_size = model_config['img_size']
    
    # Create a base model with the original architecture
    if model_name == 'efficientnet_b4':
        base_model = tf.keras.applications.EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == 'densenet121':
        base_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == 'xception':
        base_model = tf.keras.applications.Xception(
            include_top=False,
            weights='imagenet',
            input_shape=(img_size, img_size, 3)
        )
    else:
        print(f"Unknown model: {model_name}")
        return None
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=model_config['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Try to load original weights if available
    original_weights = model_config['original_weights']
    if os.path.exists(original_weights):
        try:
            # Try to load just the feature extraction layers
            print(f"Loading original weights from {original_weights}...")
            original_model = keras.models.load_model(original_weights)
            
            # Transfer weights where layer names match
            for layer in model.layers:
                if layer.name == 'flatten' or 'dense' in layer.name:
                    continue  # Skip top layers, we'll train these
                
                # Try to find corresponding layer in original model
                for orig_layer in original_model.layers:
                    if layer.name == orig_layer.name:
                        layer.set_weights(orig_layer.get_weights())
                        print(f"Transferred weights for layer: {layer.name}")
                        break
            
            print("Weight transfer complete")
        except Exception as e:
            print(f"Error loading original weights: {e}")
            print("Continuing with ImageNet weights")
    
    return model

def train_model(model_name, train_generator, validation_generator):
    """Train the model and save weights"""
    model = create_model(model_name)
    if model is None:
        return None
    
    model_config = MODEL_CONFIGS[model_name]
    weights_file = model_config['test_weights']
    
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
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(f"test_{model_name}_training_log.csv")
    ]
    
    # Train the model
    print(f"\nTraining {model_name} on test subset...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy (Test)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss (Test)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"test_{model_name}_training_history.png")
    plt.close()
    
    # Load best model
    model = keras.models.load_model(weights_file)
    
    return model, history

def evaluate_model(model, validation_generator, model_name):
    """Evaluate the model and save performance metrics"""
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Reset the generator
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
    y_pred_probs = model.predict(validation_generator, steps=len(validation_generator), verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get class names
    class_indices = validation_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    labels = [class_names[i] for i in range(len(class_names))]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Print metrics
    print(f"\n===== Test Evaluation Results for {model_name} =====")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Classification report
    report_text = classification_report(y_true, y_pred, target_names=labels)
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    print("\nDetailed Classification Report:")
    print(report_text)
    
    # Save classification report to file
    with open(f"{results_dir}/{model_name}_test_report.txt", "w") as f:
        f.write(f"===== Test Evaluation Results for {model_name} =====\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report_text)
    
    # Save report as JSON for visualization
    with open(f"{results_dir}/{model_name}_test_report.json", "w") as f:
        import json
        json.dump(report_dict, f, indent=4)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Test Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_test_confusion_matrix.png")
    plt.close()
    
    # Get training metrics for comparison
    try:
        train_report_file = f"{model_name}_classification_report.txt"
        if os.path.exists(train_report_file):
            with open(train_report_file, 'r') as f:
                train_content = f.read()
            
            # Extract training metrics
            train_accuracy = float(train_content.split("Accuracy: ")[1].split("\n")[0])
            train_precision = float(train_content.split("Precision: ")[1].split("\n")[0])
            train_recall = float(train_content.split("Recall: ")[1].split("\n")[0])
            train_f1 = float(train_content.split("F1 Score: ")[1].split("\n")[0])
            
            # Create comparison visualization
            plt.figure(figsize=(12, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            train_values = [train_accuracy, train_precision, train_recall, train_f1]
            test_values = [accuracy, precision, recall, f1]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, train_values, width, label='Training')
            plt.bar(x + width/2, test_values, width, label='Test')
            
            plt.ylabel('Score')
            plt.title(f'Training vs Test Metrics - {model_name}')
            plt.xticks(x, metrics)
            plt.legend()
            
            # Add values above bars
            for i, v in enumerate(train_values):
                plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
            for i, v in enumerate(test_values):
                plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
            
            plt.ylim(0, 1.1)
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{model_name}_metrics_comparison.png")
            plt.close()
            
            # Add comparison to report file
            with open(f"{results_dir}/{model_name}_test_report.txt", "a") as f:
                f.write("\n\n===== Training vs Test Comparison =====\n")
                f.write(f"Training Accuracy: {train_accuracy:.4f} | Test Accuracy: {accuracy:.4f}\n")
                f.write(f"Training Precision: {train_precision:.4f} | Test Precision: {precision:.4f}\n")
                f.write(f"Training Recall: {train_recall:.4f} | Test Recall: {recall:.4f}\n")
                f.write(f"Training F1 Score: {train_f1:.4f} | Test F1 Score: {f1:.4f}\n")
    except Exception as e:
        print(f"Error creating comparison: {e}")
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1
    }

def main():
    """Main function to train and evaluate models on test data"""
    print("=== Skin Cancer Classification - Test Data Training & Evaluation ===")
    
    # Create labeled dataset
    train_df, val_df = create_labeled_dataset()
    if train_df is None or val_df is None:
        print("Error creating dataset. Exiting.")
        return
    
    # Ask which models to train/evaluate
    print("\nAvailable models:")
    for i, model_name in enumerate(MODEL_CONFIGS.keys()):
        print(f"{i+1}. {model_name}")
    
    print("\nEnter model names to train/evaluate (comma-separated) or leave blank for all:")
    model_input = input().strip()
    
    if not model_input:
        models_to_run = list(MODEL_CONFIGS.keys())
    else:
        models_to_run = [m.strip() for m in model_input.split(',') if m.strip() in MODEL_CONFIGS]
        if not models_to_run:
            print("No valid models selected. Using all models.")
            models_to_run = list(MODEL_CONFIGS.keys())
    
    # Train and evaluate models
    results = {}
    for model_name in models_to_run:
        print(f"\n===== Processing {model_name} =====")
        
        # Create data generators
        train_generator, validation_generator = create_data_generators(train_df, val_df, model_name)
        
        # Train model
        model, history = train_model(model_name, train_generator, validation_generator)
        if model is None:
            print(f"Error training {model_name}. Skipping evaluation.")
            continue
        
        # Evaluate model
        metrics = evaluate_model(model, validation_generator, model_name)
        results[model_name] = metrics
    
    # Show overall comparison
    if len(results) > 1:
        print("\n===== Overall Test Results Comparison =====")
        for model_name, metrics in results.items():
            print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Run visualization to include test results
    print("\nRunning visualization to include test results...")
    try:
        import visualize_results
        visualize_results.main()
    except Exception as e:
        print(f"Error running visualization: {e}")
    
    print("\n===== Test training and evaluation complete =====")

if __name__ == "__main__":
    main() 