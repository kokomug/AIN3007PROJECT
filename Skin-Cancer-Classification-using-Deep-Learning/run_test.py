import test_models

if __name__ == "__main__":
    print("=== Running tests for Xception and DenseNet121 models ===")
    
    # Test Xception
    print("\n===== Testing xception =====")
    xception_results = test_models.evaluate_model("xception")
    
    # Test DenseNet121
    print("\n===== Testing densenet121 =====")
    densenet_results = test_models.evaluate_model("densenet121")
    
    # Run visualization to include test results
    print("\n===== Running visualization with test results =====")
    import visualize_results
    visualize_results.main()
    
    print("\n===== Testing complete =====") 