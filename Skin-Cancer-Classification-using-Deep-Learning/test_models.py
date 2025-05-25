import pandas as pd
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
TEST_CSV_PATH = 'Data/Processed CSV\'s/test_2020_withPateintDetail.csv'
# Alternative path if needed
ALT_TEST_CSV_PATH = 'Data/Processed CSV\'s/test_2020_no_PateintDetail.csv'
IMAGE_BASE_DIR = '512x512'
TEST_SUBSET_SIZE = 0.15  # Use 15% of test data
BATCH_SIZE = 16
NUM_CLASSES = 8  # Match the model output classes

# Class indices mapping (based on training data)
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

# Reverse mapping for display
CLASS_NAMES = {v: k for k, v in CLASS_INDICES.items()}

# Model configurations with preprocessing functions
MODEL_CONFIGS = {
    'efficientnet_b4': {
        'weights_file': 'efficientnet_b4_subset_best.keras',
        'img_size': 380,
        'preprocess_fn': efficientnet_preprocess
    },
    'densenet121': {
        'weights_file': 'densenet121_best.keras',
        'img_size': 224,
        'preprocess_fn': densenet_preprocess
    },
    'xception': {
        'weights_file': 'xception_best.keras',
        'img_size': 299,
        'preprocess_fn': xception_preprocess
    },
    'efficientnet_b0': {
        'weights_file': 'efficientnet_b0_best.keras',
        'img_size': 224,
        'preprocess_fn': efficientnet_preprocess
    }
}

def load_and_preprocess_test_data(model_name):
    """Load and preprocess test data for a specific model"""
    model_config = MODEL_CONFIGS[model_name]
    img_size = model_config['img_size']
    preprocess_fn = model_config['preprocess_fn']
    
    print("=== Starting Test Data Loading Process ===")
    
    # Try loading the test CSV
    if os.path.exists(TEST_CSV_PATH):
        test_csv_path = TEST_CSV_PATH
    elif os.path.exists(ALT_TEST_CSV_PATH):
        test_csv_path = ALT_TEST_CSV_PATH
    else:
        print(f"Error: Test CSV file not found")
        sys.exit(1)
        
    if not os.path.isdir(IMAGE_BASE_DIR):
        print(f"Error: Image directory '{IMAGE_BASE_DIR}' not found")
        sys.exit(1)
    
    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded {len(test_df)} total test images")
    
    # Create subset
    if TEST_SUBSET_SIZE < 1.0:
        print(f"\nCreating {TEST_SUBSET_SIZE*100}% test subset...")
        test_subset, _ = train_test_split(test_df, 
                                      train_size=TEST_SUBSET_SIZE,
                                      random_state=42)
        test_df = test_subset
        print(f"Test subset size: {len(test_df)} images")
    
    # Add image paths
    test_df['image_path'] = test_df['image'].apply(lambda x: os.path.join(IMAGE_BASE_DIR, f'{x}.jpg'))
    
    # Check if all image paths exist
    missing_images = [path for path in test_df['image_path'] if not os.path.exists(path)]
    if missing_images:
        print(f"Warning: {len(missing_images)} images not found in {IMAGE_BASE_DIR}")
        print(f"First few missing: {missing_images[:5]}")
        test_df = test_df[~test_df['image_path'].isin(missing_images)]
        print(f"Proceeding with {len(test_df)} images that were found")
    
    # Create test data generator without labels
    print("\nSetting up test data generator...")
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn
    )
    
    # Check if test data has labels
    if 'diagnosis' in test_df.columns:
        print("Using labeled test data")
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='image_path',
            y_col='diagnosis',
            target_size=(img_size, img_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        has_labels = True
    else:
        print("Using unlabeled test data")
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='image_path',
            y_col=None,
            target_size=(img_size, img_size),
            batch_size=BATCH_SIZE,
            class_mode=None,
            shuffle=False
        )
        has_labels = False
    
    return test_generator, test_df, has_labels

def evaluate_model(model_name):
    """Evaluate a trained model on test data"""
    if model_name not in MODEL_CONFIGS:
        print(f"Error: Unknown model {model_name}")
        return
    
    model_config = MODEL_CONFIGS[model_name]
    weights_file = model_config['weights_file']
    
    # Check if model exists
    if not os.path.exists(weights_file):
        print(f"Error: Model weights file not found at {weights_file}")
        return
    
    # Load the model
    print(f"\nLoading model: {model_name}")
    model = keras.models.load_model(weights_file)
    
    # Load test data
    test_generator, test_df, has_labels = load_and_preprocess_test_data(model_name)
    
    # Evaluate model
    print(f"\nEvaluating model: {model_name} on test data...")
    
    # Get predictions
    y_pred_probs = model.predict(test_generator, steps=None, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get class names for display
    labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    
    # Generate confusion matrix from validation data if no test labels
    if not has_labels:
        print("\nNo ground truth labels available for test data.")
        print("Using predictions only for analysis.")
        
        # Count predictions by class
        print("\nPrediction distribution:")
        pred_counts = np.bincount(y_pred, minlength=NUM_CLASSES)
        for i, count in enumerate(pred_counts):
            if i < len(labels):
                print(f"{labels[i]}: {count} ({count/len(y_pred)*100:.1f}%)")
        
        # Get validation confusion matrix if available (for visualization)
        val_cm_paths = glob.glob("*confusion_matrix.png")
        if val_cm_paths:
            print(f"\nValidation confusion matrix available at: {val_cm_paths[0]}")
        
        # Save predictions
        results_dir = "test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare prediction data
        pred_df = pd.DataFrame({
            'image': test_df['image'],
            'predicted_class': [labels[i] for i in y_pred],
            'predicted_class_id': y_pred
        })
        
        # Add probabilities for each class
        for i, class_name in enumerate(labels):
            pred_df[f'prob_{class_name}'] = y_pred_probs[:, i]
        
        # Save predictions
        pred_df.to_csv(f"{results_dir}/{model_name}_test_predictions.csv", index=False)
        print(f"\nPredictions saved to {results_dir}/{model_name}_test_predictions.csv")
        
        # Create prediction distribution pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(pred_counts, labels=labels, autopct='%1.1f%%', shadow=True)
        plt.title(f'Prediction Distribution - {model_name}')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{model_name}_prediction_distribution.png")
        plt.close()
        
        # Generate sample report (using prediction distribution)
        report = {
            "accuracy": None,  # No ground truth, so no accuracy
            "macro avg": {
                "precision": None,
                "recall": None,
                "f1-score": None,
                "support": len(y_pred)
            },
            "weighted avg": {
                "precision": None,
                "recall": None,
                "f1-score": None,
                "support": len(y_pred)
            }
        }
        
        # Add class distribution to report
        for i, label in enumerate(labels):
            report[label] = {
                "precision": None,
                "recall": None,
                "f1-score": None,
                "support": int(pred_counts[i])
            }
        
        # Save report as JSON for visualization
        with open(f"{results_dir}/{model_name}_test_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        # Create a text report
        with open(f"{results_dir}/{model_name}_test_report.txt", "w") as f:
            f.write(f"===== Test Evaluation Results for {model_name} =====\n")
            f.write("NOTE: No ground truth labels available. These are prediction statistics only.\n\n")
            f.write("Prediction Distribution:\n")
            for i, count in enumerate(pred_counts):
                if i < len(labels):
                    f.write(f"{labels[i]}: {count} ({count/len(y_pred)*100:.1f}%)\n")
        
        print(f"Test results saved to {results_dir}/{model_name}_test_report.txt")
        return None, report, None
    
    # If we have ground truth labels, proceed with regular evaluation
    # Get true labels
    y_true = []
    batch_count = 0
    for _, labels in test_generator:
        y_true.extend(np.argmax(labels, axis=1))
        batch_count += 1
        if batch_count >= len(test_generator):
            break
    
    # If y_true and y_pred have different lengths, trim the longer one
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=labels)
    
    print(f"\n===== Test Evaluation Results for {model_name} =====")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(report_text)
    
    # Save results
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save report
    with open(f"{results_dir}/{model_name}_test_report.txt", "w") as f:
        f.write(f"===== Test Evaluation Results for {model_name} =====\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report_text)
    
    # Save report as JSON for easier parsing
    with open(f"{results_dir}/{model_name}_test_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Test Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_test_confusion_matrix.png")
    plt.close()
    
    # Save raw predictions for later analysis
    pred_df = pd.DataFrame({
        'image': test_df['image'].values[:min_len],
        'true_class': [labels[i] for i in y_true],
        'pred_class': [labels[i] for i in y_pred],
        'correct': [y_true[i] == y_pred[i] for i in range(len(y_true))]
    })
    
    # Add probabilities for each class
    for i, class_name in enumerate(labels):
        pred_df[f'prob_{class_name}'] = y_pred_probs[:min_len, i]
    
    pred_df.to_csv(f"{results_dir}/{model_name}_test_predictions.csv", index=False)
    
    print(f"\nTest results saved to {results_dir}/{model_name}_test_report.txt")
    print(f"Test confusion matrix saved to {results_dir}/{model_name}_test_confusion_matrix.png")
    
    return accuracy, report, cm

def main():
    """Main function for testing models"""
    print("=== Skin Cancer Classification - Model Testing ===")
    
    # Check which models to evaluate
    available_models = []
    for model_name, config in MODEL_CONFIGS.items():
        if os.path.exists(config['weights_file']):
            available_models.append(model_name)
    
    if not available_models:
        print("No trained models found. Please train models first.")
        return
    
    print(f"\nAvailable trained models: {available_models}")
    
    # Ask which model to test
    print("\nEnter the number or name of the model to test")
    print("(or leave blank to test all available models):")
    model_input = input().strip()
    
    models_to_test = []
    if not model_input:
        # Test all available models
        models_to_test = available_models
    elif model_input.isdigit():
        # Test by number
        idx = int(model_input) - 1
        if 0 <= idx < len(available_models):
            models_to_test = [available_models[idx]]
        else:
            print(f"Invalid model number. Will test all models.")
            models_to_test = available_models
    else:
        # Test by name
        if model_input in available_models:
            models_to_test = [model_input]
        else:
            print(f"Model {model_input} not found or not trained. Will test all models.")
            models_to_test = available_models
    
    # Test selected models
    results = {}
    for model_name in models_to_test:
        print(f"\n===== Testing {model_name} =====")
        accuracy, report, _ = evaluate_model(model_name)
        results[model_name] = {
            'accuracy': accuracy,
            'report': report
        }
    
    # Compare results if multiple models tested
    if len(models_to_test) > 1:
        print("\n===== Test Results Comparison =====")
        for model_name in models_to_test:
            acc = results[model_name]['accuracy']
            if acc is not None:
                print(f"{model_name}: Test Accuracy = {acc:.4f}")
            else:
                print(f"{model_name}: No accuracy (unlabeled test data)")
    
    print("\n===== Testing complete =====")

if __name__ == "__main__":
    main() 