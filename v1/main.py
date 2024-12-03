# main.py

def main():
    # Import necessary modules
    from preProcessing.data_preprocessing import preprocess_meteorological_data
    from preProcessing.image_preprocessing import preprocess_images
    from training.train_model import train_hybrid_model
    from evaluation.evaluate_model import evaluate_model

    # Step 1: Preprocess Data
    met_data = preprocess_meteorological_data('data/meteorological_data.csv')
    images, image_timestamps = preprocess_images('data/images/')

    # Step 2: Train Model
    model, history, merged_data = train_hybrid_model(met_data, images, image_timestamps)

    # Step 3: Evaluate Model
    evaluate_model(model, merged_data)

if __name__ == '__main__':
    main()