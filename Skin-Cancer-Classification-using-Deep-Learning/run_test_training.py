import train_test_evaluation

if __name__ == "__main__":
    print("=== Running test training and evaluation for Xception and DenseNet121 models ===")
    
    # Create labeled dataset
    train_df, val_df = train_test_evaluation.create_labeled_dataset()
    if train_df is None or val_df is None:
        print("Error creating dataset. Exiting.")
        exit(1)
    
    # Process Xception
    print("\n===== Processing xception =====")
    train_generator, validation_generator = train_test_evaluation.create_data_generators(
        train_df, val_df, "xception"
    )
    model, history = train_test_evaluation.train_model("xception", train_generator, validation_generator)
    if model is not None:
        metrics = train_test_evaluation.evaluate_model(model, validation_generator, "xception")
    
    # Process DenseNet121
    print("\n===== Processing densenet121 =====")
    train_generator, validation_generator = train_test_evaluation.create_data_generators(
        train_df, val_df, "densenet121"
    )
    model, history = train_test_evaluation.train_model("densenet121", train_generator, validation_generator)
    if model is not None:
        metrics = train_test_evaluation.evaluate_model(model, validation_generator, "densenet121")
    
    # Run visualization to include test results
    print("\n===== Running visualization with test results =====")
    import visualize_results
    visualize_results.main()
    
    print("\n===== Test training and evaluation complete =====") 