import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
import glob
import json
from sklearn.metrics import confusion_matrix
import random

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
IMAGE_DIR = '512x512'
TEST_SUBSET_SIZE = 0.15  # Use 15% of test data
BATCH_SIZE = 16
NUM_CLASSES = 8

# Class names for reference
CLASS_NAMES = [
    'MEL',  # Melanoma
    'NV',   # Melanocytic nevus
    'BCC',  # Basal cell carcinoma
    'AK',   # Actinic keratosis
    'BKL',  # Benign keratosis
    'DF',   # Dermatofibroma
    'VASC', # Vascular lesion
    'SCC'   # Squamous cell carcinoma
]

# Model configurations
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
    }
}

def create_sample_dataset():
    """Create a sample dataset from available images"""
    print("Creating sample dataset for testing...")
    
    # Get all image files
    image_files = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    
    if not image_files:
        print(f"Error: No images found in {IMAGE_DIR}")
        return None
    
    print(f"Found {len(image_files)} images")
    
    # Select a random sample (15% of images)
    sample_size = int(len(image_files) * TEST_SUBSET_SIZE)
    if len(image_files) > sample_size:
        sample_files = random.sample(image_files, sample_size)
    else:
        sample_files = image_files
    
    # Create a dataframe with just image paths
    sample_df = pd.DataFrame({
        'image': [os.path.basename(f).replace('.jpg', '') for f in sample_files],
        'image_path': sample_files
    })
    
    print(f"Created sample dataset with {len(sample_df)} images ({TEST_SUBSET_SIZE*100:.1f}% of available data)")
    return sample_df

def get_training_metrics(model_name):
    """Get training metrics from saved classification report if available"""
    report_file = f"{model_name}_classification_report.txt"
    
    metrics = {
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None
    }
    
    if os.path.exists(report_file):
        try:
            with open(report_file, 'r') as f:
                content = f.read()
                
            # Extract metrics
            for metric in metrics.keys():
                match = None
                if metric == 'accuracy':
                    match = content.find(f"Accuracy: ")
                else:
                    match = content.find(f"{metric.capitalize()}: ")
                
                if match >= 0:
                    start = match + len(f"{metric.capitalize()}: ")
                    end = content.find("\n", start)
                    value = content[start:end].strip()
                    try:
                        metrics[metric] = float(value)
                    except ValueError:
                        pass
            
            print(f"Loaded training metrics for {model_name}")
        except Exception as e:
            print(f"Error loading training metrics for {model_name}: {e}")
    
    return metrics

def test_model(model_name):
    """Test a model on sample images"""
    if model_name not in MODEL_CONFIGS:
        print(f"Error: Unknown model {model_name}")
        return None
    
    # Get model configuration
    model_config = MODEL_CONFIGS[model_name]
    weights_file = model_config['weights_file']
    img_size = model_config['img_size']
    preprocess_fn = model_config['preprocess_fn']
    
    # Check if model exists
    if not os.path.exists(weights_file):
        print(f"Error: Model weights file not found at {weights_file}")
        return None
    
    # Create sample dataset
    sample_df = create_sample_dataset()
    if sample_df is None:
        return None
    
    # Load the model
    print(f"\nLoading model: {model_name}")
    model = keras.models.load_model(weights_file)
    
    # Setup data generator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    
    # Create generator
    generator = datagen.flow_from_dataframe(
        dataframe=sample_df,
        x_col="image_path",
        y_col=None,
        class_mode=None,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Generate predictions
    print(f"Generating predictions for {len(sample_df)} images...")
    predictions = model.predict(generator, steps=len(generator), verbose=1)
    
    # Process predictions
    pred_classes = np.argmax(predictions, axis=1)
    
    # Print distribution of predictions
    print("\nPrediction distribution:")
    pred_counts = {}
    for i, class_name in enumerate(CLASS_NAMES):
        count = np.sum(pred_classes == i)
        percentage = count/len(pred_classes)*100
        pred_counts[class_name] = count
        print(f"{class_name}: {count} ({percentage:.1f}%)")
    
    # Get training metrics for comparison
    training_metrics = get_training_metrics(model_name)
    
    # Save results
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'image': sample_df['image'],
        'predicted_class': [CLASS_NAMES[i] for i in pred_classes],
        'predicted_class_id': pred_classes
    })
    
    # Add prediction probabilities
    for i, class_name in enumerate(CLASS_NAMES):
        pred_df[f'prob_{class_name}'] = predictions[:, i]
    
    # Save predictions
    pred_df.to_csv(f"{results_dir}/{model_name}_test_predictions.csv", index=False)
    
    # Create prediction distribution chart (styled like confusion matrix)
    plt.figure(figsize=(10, 8))
    
    # Create a fake confusion matrix just to show prediction distribution
    # This makes it visually similar to the training confusion matrix
    fake_cm = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        fake_cm[i, i] = pred_counts.get(CLASS_NAMES[i], 0)
    
    sns.heatmap(fake_cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Predicted (No Ground Truth)')
    plt.title(f'Test Prediction Distribution - {model_name}')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_test_prediction_matrix.png")
    plt.close()
    
    # Create regular distribution chart
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=CLASS_NAMES, y=[pred_counts.get(c, 0) for c in CLASS_NAMES])
    plt.title(f"Test Prediction Distribution - {model_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    # Add counts above bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                f"{int(height)}\n({int(height)/len(pred_classes)*100:.1f}%)",
                ha = 'center')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model_name}_test_prediction_dist.png")
    plt.close()
    
    # Generate a report structure similar to training metrics report
    # But note that actual metrics cannot be calculated without ground truth
    report = {
        "accuracy": None,  # No ground truth
        "precision": None,
        "recall": None,
        "f1-score": None,
        "support": len(pred_classes),
        "macro avg": {
            "precision": None,
            "recall": None,
            "f1-score": None,
            "support": len(pred_classes)
        },
        "weighted avg": {
            "precision": None,
            "recall": None,
            "f1-score": None,
            "support": len(pred_classes)
        }
    }
    
    # Add class distribution to report
    for i, class_name in enumerate(CLASS_NAMES):
        count = np.sum(pred_classes == i)
        report[class_name] = {
            "precision": None,
            "recall": None,
            "f1-score": None,
            "support": int(count)
        }
    
    # Save report as JSON for visualization
    with open(f"{results_dir}/{model_name}_test_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Create report text in the same format as training reports
    with open(f"{results_dir}/{model_name}_test_report.txt", "w") as f:
        f.write(f"===== Test Evaluation Results for {model_name} =====\n")
        f.write(f"NOTE: This is a test on {len(sample_df)} images ({TEST_SUBSET_SIZE*100:.1f}% of available data) without ground truth labels.\n\n")
        f.write("Test Prediction Summary (No Ground Truth):\n")
        
        # Add placeholder metrics for compatibility with visualize_results.py
        f.write("Accuracy: N/A\n")
        f.write("Precision: N/A\n")
        f.write("Recall: N/A\n")
        f.write("F1 Score: N/A\n\n")
        
        # Add prediction distribution
        f.write("Prediction Distribution:\n")
        for i, class_name in enumerate(CLASS_NAMES):
            count = np.sum(pred_classes == i)
            f.write(f"{class_name}: {count} ({count/len(pred_classes)*100:.1f}%)\n")
        
        # Add comparison with training metrics if available
        if any(training_metrics.values()):
            f.write("\nTraining Metrics (For Comparison):\n")
            for metric, value in training_metrics.items():
                if value is not None:
                    f.write(f"{metric.capitalize()}: {value:.4f}\n")
    
    print(f"\nSaved test results to {results_dir}/{model_name}_test_report.txt")
    print(f"Saved distribution charts to {results_dir}/{model_name}_test_prediction_dist.png")
    
    # Generate comparison plot between training and test prediction distribution
    try:
        # Try to load training confusion matrix
        cm_file = f"{model_name}_confusion_matrix.png"
        if os.path.exists(cm_file):
            print(f"Creating training/test visualization comparison...")
            
            # Create a simple comparison visualization
            plt.figure(figsize=(16, 8))
            plt.suptitle(f"Training vs Test Distribution - {model_name}", fontsize=16)
            
            # Display training metrics
            plt.subplot(121)
            plt.text(0.5, 0.5, f"Training Metrics:\nAccuracy: {training_metrics.get('accuracy', 'N/A'):.4f}\n"
                              f"Precision: {training_metrics.get('precision', 'N/A'):.4f}\n"
                              f"Recall: {training_metrics.get('recall', 'N/A'):.4f}\n"
                              f"F1 Score: {training_metrics.get('f1', 'N/A'):.4f}",
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            
            # Display test prediction distribution
            plt.subplot(122)
            sns.barplot(x=CLASS_NAMES, y=[pred_counts.get(c, 0) for c in CLASS_NAMES])
            plt.title("Test Prediction Distribution")
            plt.xlabel("Predicted Class")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{model_name}_train_test_comparison.png")
            plt.close()
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")
    
    return pred_df, predictions

def main():
    """Run test on models"""
    print("=== Skin Cancer Classification - Sample Testing ===")
    
    # Check available models
    available_models = []
    for model_name, config in MODEL_CONFIGS.items():
        if os.path.exists(config['weights_file']):
            available_models.append(model_name)
    
    if not available_models:
        print("No trained models found. Please train models first.")
        return
    
    print(f"Available models: {available_models}")
    
    # Ask which models to test
    print("\nEnter model names to test (comma-separated) or leave blank for all:")
    model_input = input().strip()
    
    if not model_input:
        models_to_test = available_models
    else:
        models_to_test = [m.strip() for m in model_input.split(',') if m.strip() in available_models]
        if not models_to_test:
            print("No valid models selected. Testing all available models.")
            models_to_test = available_models
    
    # Test models
    results = {}
    for model_name in models_to_test:
        print(f"\n===== Testing {model_name} =====")
        results[model_name] = test_model(model_name)
    
    # Run visualization to integrate results
    print("\nRunning visualization to include test results...")
    try:
        import visualize_results
        visualize_results.main()
    except Exception as e:
        print(f"Error running visualization: {e}")
    
    print("\n===== Testing complete =====")

if __name__ == "__main__":
    main()