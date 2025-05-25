import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import re
import json

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

def load_training_logs(model_names):
    """Load training logs for specified models"""
    histories = {}
    for model_name in model_names:
        log_file = f"{model_name}_training_log.csv"
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                histories[model_name] = df
                print(f"Loaded training log for {model_name}")
            except Exception as e:
                print(f"Error loading log for {model_name}: {e}")
        else:
            print(f"Training log for {model_name} not found")
    
    return histories

def extract_metrics_from_reports(model_names):
    """Extract performance metrics from classification reports"""
    metrics = {}
    class_metrics = {}
    
    for model_name in model_names:
        report_file = f"{model_name}_classification_report.txt"
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r') as f:
                    content = f.read()
                
                # Extract overall metrics
                accuracy_match = re.search(r'Accuracy:\s+(\d+\.\d+)', content)
                precision_match = re.search(r'Precision:\s+(\d+\.\d+)', content)
                recall_match = re.search(r'Recall:\s+(\d+\.\d+)', content)
                f1_match = re.search(r'F1 Score:\s+(\d+\.\d+)', content)
                
                if accuracy_match and precision_match and recall_match and f1_match:
                    metrics[model_name] = {
                        'accuracy': float(accuracy_match.group(1)),
                        'precision': float(precision_match.group(1)),
                        'recall': float(recall_match.group(1)),
                        'f1': float(f1_match.group(1))
                    }
                
                # Extract per-class metrics
                class_metrics[model_name] = {}
                
                # Match all class metrics using regex
                class_pattern = r'\s+(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)'
                for match in re.finditer(class_pattern, content):
                    class_name, precision, recall, f1, support = match.groups()
                    class_metrics[model_name][class_name] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'support': int(support)
                    }
                
                print(f"Extracted metrics for {model_name}")
            except Exception as e:
                print(f"Error extracting metrics for {model_name}: {e}")
        else:
            print(f"Classification report for {model_name} not found")
    
    return metrics, class_metrics

def extract_test_metrics(model_names):
    """Extract performance metrics from test classification reports"""
    test_metrics = {}
    test_class_metrics = {}
    
    for model_name in model_names:
        report_file = f"test_results/{model_name}_test_report.txt"
        json_report_file = f"test_results/{model_name}_test_report.json"
        
        # Try reading from JSON first (more reliable)
        if os.path.exists(json_report_file):
            try:
                with open(json_report_file, 'r') as f:
                    report = json.load(f)
                
                # Extract metrics from JSON
                test_metrics[model_name] = {
                    'accuracy': report['accuracy'],
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1': report['weighted avg']['f1-score']
                }
                
                # Extract per-class metrics
                test_class_metrics[model_name] = {}
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict) and 'precision' in metrics and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                        test_class_metrics[model_name][class_name] = {
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1-score'],
                            'support': metrics['support']
                        }
                
                print(f"Extracted test metrics for {model_name} from JSON")
                continue
            except Exception as e:
                print(f"Error extracting test metrics from JSON for {model_name}: {e}")
        
        # Fall back to text file if JSON not available
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r') as f:
                    content = f.read()
                
                # Extract overall metrics
                accuracy_match = re.search(r'Test Accuracy:\s+(\d+\.\d+)', content)
                
                if accuracy_match:
                    test_metrics[model_name] = {
                        'accuracy': float(accuracy_match.group(1))
                    }
                
                    # Try to extract precision, recall, f1 from weighted avg
                    weighted_pattern = r'weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
                    weighted_match = re.search(weighted_pattern, content)
                    if weighted_match:
                        precision, recall, f1 = weighted_match.groups()
                        test_metrics[model_name].update({
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1': float(f1)
                        })
                
                # Extract per-class metrics
                test_class_metrics[model_name] = {}
                
                # Match all class metrics using regex
                class_pattern = r'\s+(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)'
                for match in re.finditer(class_pattern, content):
                    class_name, precision, recall, f1, support = match.groups()
                    test_class_metrics[model_name][class_name] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'support': int(support)
                    }
                
                print(f"Extracted test metrics for {model_name} from text file")
            except Exception as e:
                print(f"Error extracting test metrics from text file for {model_name}: {e}")
        else:
            print(f"Test report for {model_name} not found")
    
    return test_metrics, test_class_metrics

def plot_training_history(histories, save_path='training_history_comparison.png'):
    """Plot training histories for multiple models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for model_name, history in histories.items():
        if 'accuracy' in history.columns:
            axes[0].plot(history['epoch'], history['accuracy'], marker='o', markersize=3, 
                         label=f'{model_name} (Train)')
        if 'val_accuracy' in history.columns:
            axes[0].plot(history['epoch'], history['val_accuracy'], marker='x', markersize=3, 
                         linestyle='--', label=f'{model_name} (Val)')
    
    for model_name, history in histories.items():
        if 'loss' in history.columns:
            axes[1].plot(history['epoch'], history['loss'], marker='o', markersize=3, 
                         label=f'{model_name} (Train)')
        if 'val_loss' in history.columns:
            axes[1].plot(history['epoch'], history['val_loss'], marker='x', markersize=3, 
                         linestyle='--', label=f'{model_name} (Val)')
    
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc='lower right')
    axes[0].grid(True)
    
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training history plot to {save_path}")

def plot_overall_metrics(metrics, test_metrics=None, save_path='overall_metrics_comparison.png'):
    """Plot overall metrics comparison"""
    metrics_df = pd.DataFrame(metrics).T
    
    # Setup the plot
    if test_metrics:
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot validation metrics
    metrics_df.plot(kind='bar', ax=ax1)
    ax1.set_title('Validation Performance Metrics Comparison')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.legend(title='Metric')
    ax1.grid(axis='y')
    
    # Add value labels on the bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f', padding=3)
    
    # Plot test metrics if available
    if test_metrics:
        test_metrics_df = pd.DataFrame(test_metrics).T
        test_metrics_df.plot(kind='bar', ax=ax2, color='orange')
        ax2.set_title('Test Performance Metrics Comparison')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        ax2.legend(title='Metric')
        ax2.grid(axis='y')
        
        # Add value labels on the bars
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved overall metrics plot to {save_path}")

def plot_class_f1_comparison(class_metrics, test_class_metrics=None, save_path='class_f1_comparison.png'):
    """Plot F1 scores per class across models"""
    # Extract F1 scores for each class and model
    classes = set()
    for model_metrics in class_metrics.values():
        classes.update(model_metrics.keys())
    
    # Check if test metrics are available
    test_classes = set()
    if test_class_metrics:
        for model_metrics in test_class_metrics.values():
            test_classes.update(model_metrics.keys())
        classes.update(test_classes)
    
    classes = sorted(list(classes))
    models = list(class_metrics.keys())
    
    # Create data for validation plot
    data = []
    for model in models:
        if model in class_metrics:
            for class_name in classes:
                if class_name in class_metrics[model]:
                    data.append({
                        'Model': model,
                        'Class': class_name,
                        'F1 Score': class_metrics[model][class_name]['f1'],
                        'Dataset': 'Validation'
                    })
    
    # Add test data if available
    if test_class_metrics:
        for model in models:
            if model in test_class_metrics:
                for class_name in classes:
                    if class_name in test_class_metrics[model]:
                        data.append({
                            'Model': model,
                            'Class': class_name,
                            'F1 Score': test_class_metrics[model][class_name]['f1'],
                            'Dataset': 'Test'
                        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Determine how to plot based on data
    if 'Dataset' in df.columns and len(df['Dataset'].unique()) > 1:
        # Plot with validation and test side by side
        g = sns.catplot(
            data=df, 
            kind="bar",
            x="Class", y="F1 Score", hue="Model", col="Dataset",
            height=8, aspect=1.5, palette="deep"
        )
        g.fig.suptitle('F1 Score by Class and Model', fontsize=16)
        g.fig.subplots_adjust(top=0.85)
        g.set_axis_labels('Class', 'F1 Score')
        g.set_xticklabels(rotation=45)
        plt.savefig(save_path)
        plt.close()
    else:
        # Original single plot
        plt.figure(figsize=(14, 8))
        chart = sns.barplot(x='Class', y='F1 Score', hue='Model', data=df)
        
        chart.set_title('F1 Score by Class and Model')
        chart.set_xlabel('Class')
        chart.set_ylabel('F1 Score')
        chart.set_ylim(0, 1)
        chart.legend(title='Model')
        chart.grid(axis='y')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    print(f"Saved class F1 comparison plot to {save_path}")

def plot_model_size_vs_performance(metrics, test_metrics, model_sizes, save_path='model_size_vs_performance.png'):
    """Plot model size vs performance metrics for both validation and test"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = list(metrics.keys())
    x = [model_sizes[model] for model in models if model in model_sizes]
    
    # Validation metrics
    y_val_accuracy = [metrics[model]['accuracy'] for model in models if model in model_sizes]
    
    # Test metrics if available
    if test_metrics:
        y_test_accuracy = [test_metrics[model]['accuracy'] for model in models 
                          if model in model_sizes and model in test_metrics]
        
        # Only include models with both test and validation data
        models_with_both = [m for m in models if m in model_sizes and m in test_metrics]
        x_both = [model_sizes[m] for m in models_with_both]
        y_val_both = [metrics[m]['accuracy'] for m in models_with_both]
        y_test_both = [test_metrics[m]['accuracy'] for m in models_with_both]
        
        # Create scatter plot for both
        ax.scatter(x_both, y_val_both, s=100, alpha=0.7, label='Validation')
        ax.scatter(x_both, y_test_both, s=100, alpha=0.7, marker='d', label='Test')
        
        # Connect validation and test points for the same model
        for i, model in enumerate(models_with_both):
            ax.plot([x_both[i], x_both[i]], [y_val_both[i], y_test_both[i]], 
                    'k-', alpha=0.5, linewidth=1)
    else:
        # Just validation metrics
        ax.scatter(x, y_val_accuracy, s=100, alpha=0.7, label='Validation')
    
    # Add labels for each point
    for i, model in enumerate(models):
        if model in model_sizes:
            idx = x.index(model_sizes[model])
            ax.annotate(model, (x[idx], y_val_accuracy[idx]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10)
    
    ax.set_title('Model Size vs. Accuracy')
    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved model size vs performance plot to {save_path}")

def create_parameter_comparison(model_configs, save_path='parameter_comparison.png'):
    """Create a parameter count comparison visualization"""
    models = list(model_configs.keys())
    params = [model_configs[model]['parameters'] for model in models]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, params, color=sns.color_palette('deep', len(models)))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}M',
                ha='center', va='bottom', fontsize=10)
    
    plt.title('Model Parameter Count Comparison')
    plt.ylabel('Parameters (millions)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved parameter comparison plot to {save_path}")

def plot_validation_vs_test(metrics, test_metrics, save_path='validation_vs_test.png'):
    """Plot validation vs test accuracy comparison"""
    if not test_metrics:
        print("No test metrics available for comparison")
        return
    
    # Find models with both validation and test metrics
    common_models = [m for m in metrics.keys() if m in test_metrics]
    
    if not common_models:
        print("No models have both validation and test metrics")
        return
    
    # Create dataframe for plotting
    data = []
    for model in common_models:
        data.append({
            'Model': model,
            'Validation Accuracy': metrics[model]['accuracy'],
            'Test Accuracy': test_metrics[model]['accuracy']
        })
    
    df = pd.DataFrame(data)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    x = np.arange(len(common_models))
    width = 0.35
    
    val_bars = ax.bar(x - width/2, df['Validation Accuracy'], width, label='Validation')
    test_bars = ax.bar(x + width/2, df['Test Accuracy'], width, label='Test')
    
    # Add labels, title and legend
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation vs Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(common_models, rotation=45)
    ax.legend()
    
    # Add value labels
    ax.bar_label(val_bars, fmt='%.3f')
    ax.bar_label(test_bars, fmt='%.3f')
    
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved validation vs test comparison to {save_path}")

def plot_confusion_matrices(model_names, save_path='confusion_matrices.png'):
    """Plot confusion matrices for multiple models"""
    # Check for test confusion matrices
    confusion_matrices = {}
    class_names = {}
    
    for model_name in model_names:
        cm_path = f"test_results/{model_name}_test_confusion_matrix.png"
        if os.path.exists(cm_path):
            # Just use the existing confusion matrix images
            confusion_matrices[model_name] = cm_path
            print(f"Found confusion matrix for {model_name}")
    
    if not confusion_matrices:
        print("No confusion matrices found")
        return
    
    # Just note the existence of the confusion matrices
    print("Confusion matrices are already available in test_results/ directory")
    print("Individual confusion matrix plots:")
    for model_name, path in confusion_matrices.items():
        print(f" - {path}")

def main():
    # Define models to compare
    models = ['efficientnet_b4', 'densenet121']
    
    # Add more models if reports exist
    for model_name in ['xception', 'efficientnet_b0']:
        if os.path.exists(f"{model_name}_classification_report.txt"):
            models.append(model_name)
    
    print(f"Comparing models: {models}")
    
    # Load training logs
    histories = load_training_logs(models)
    
    # Load metrics from classification reports
    metrics, class_metrics = extract_metrics_from_reports(models)
    
    # Check if test results are available
    test_dir = "test_results"
    has_test_results = os.path.isdir(test_dir) and any(
        os.path.exists(f"{test_dir}/{model}_test_report.txt") or 
        os.path.exists(f"{test_dir}/{model}_test_report.json") 
        for model in models
    )
    
    test_metrics = None
    test_class_metrics = None
    
    if has_test_results:
        print("Found test results. Including in visualizations.")
        test_metrics, test_class_metrics = extract_test_metrics(models)
    
    # Create visualizations
    if histories:
        plot_training_history(histories)
    
    if metrics:
        plot_overall_metrics(metrics, test_metrics)
    
    if class_metrics:
        plot_class_f1_comparison(class_metrics, test_class_metrics)
    
    if metrics and test_metrics:
        plot_validation_vs_test(metrics, test_metrics)
    
    # Plot confusion matrices
    plot_confusion_matrices(models)
    
    # Define approximate model sizes in MB
    model_sizes = {
        'efficientnet_b4': 69,
        'efficientnet_b0': 16,
        'densenet121': 32,
        'xception': 88,
        'mobilenet_v2': 14,
        'mobilenet_v3_small': 10
    }
    
    # Define parameter counts in millions
    model_configs = {
        'efficientnet_b4': {'parameters': 19.0},
        'efficientnet_b0': {'parameters': 5.3},
        'densenet121': {'parameters': 8.0},
        'xception': {'parameters': 22.9},
        'mobilenet_v2': {'parameters': 3.5},
        'mobilenet_v3_small': {'parameters': 2.5}
    }
    
    # Filter to only include trained models
    filtered_metrics = {k: v for k, v in metrics.items() if k in model_sizes}
    filtered_sizes = {k: v for k, v in model_sizes.items() if k in metrics}
    filtered_configs = {k: v for k, v in model_configs.items() if k in metrics}
    
    # Create additional visualizations
    if filtered_metrics and filtered_sizes:
        plot_model_size_vs_performance(filtered_metrics, test_metrics, filtered_sizes)
    
    if filtered_configs:
        create_parameter_comparison(filtered_configs)
    
    print("Visualization generation complete!")

if __name__ == "__main__":
    main() 