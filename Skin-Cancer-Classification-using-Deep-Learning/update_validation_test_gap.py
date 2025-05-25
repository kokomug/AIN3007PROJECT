import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

def update_validation_test_gap():
    # Models
    models = ['DenseNet121', 'Xception', 'EfficientNetB0 (limited)']
    
    # Validation accuracy
    val_accuracy = [65.51, 64.29, 60.0]
    
    # Test accuracy - Now including DenseNet121's value of 14.4%
    test_accuracy = [14.4, 13.11, None]  # Added DenseNet121 test accuracy
    
    # Create dataframe for plotting
    data = []
    for i, model in enumerate(models):
        if val_accuracy[i] is not None:
            data.append({'Model': model, 'Metric': 'Validation', 'Accuracy': val_accuracy[i]})
        if test_accuracy[i] is not None:
            data.append({'Model': model, 'Metric': 'Test', 'Accuracy': test_accuracy[i]})
    
    df = pd.DataFrame(data)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar plot
    chart = sns.barplot(x='Model', y='Accuracy', hue='Metric', data=df)
    
    # Add value labels
    for container in chart.containers:
        chart.bar_label(container, fmt='%.1f%%')
    
    # Customize plot
    plt.title('Validation vs Test Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.savefig('updated_validation_test_gap.png', bbox_inches='tight')
    plt.close()
    
    print("Created updated validation vs test gap visualization with DenseNet121 test results")

if __name__ == "__main__":
    print("Updating visualization to include DenseNet121 test accuracy...")
    update_validation_test_gap()
    print("Visualization updated successfully!") 