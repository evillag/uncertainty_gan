import tensorflow as tf
import numpy as np
from tqdm import tqdm


class MCDEvaluator:
    def __init__(self, models, datasets, num_reps, sub_sample_size):
        self.models = models
        self.datasets = datasets
        self.num_reps = num_reps
        self.sub_sample_size = sub_sample_size

    def subsample_dataset(self, features, targets, size):
        indices = np.random.choice(len(features), size, replace=False)
        return features[indices], targets[indices]

    def evaluate_model(self, model, particle, dropout):
        generator = model.get_generator()
        generator.ensemble_inference_mode()
        ensemble_inferences = dict()
        predictions_variance = []

        print(f"Generating predictions with model for particle {particle.upper()} and dropout {dropout}")
        for ensemble_size in tqdm(range(1, 11)):  # Assuming ENSEMBLES ranges from 1 to 10
            inference_predictions = dict()
            for rep in range(self.num_reps):
                try:
                    key = f'{rep}'
                    ensemble_inferences[f'{ensemble_size}'] = {key: inference_predictions}

                    # Step 1: Draw a sample of the datasets
                    x_sample, y_sample = self.subsample_dataset(self.datasets[particle]['feats_val'],
                                                                self.datasets[particle]['targets_val'],
                                                                self.sub_sample_size)

                    inference_predictions[key] = {
                        'predicted_values': None,
                        'predicted_variance': None,
                    }
                    prediction_list = []

                    for inference in range(ensemble_size):
                        prediction_list.append(generator(x_sample))

                    predicted_values = tf.convert_to_tensor(prediction_list)

                    # Step 2: Calculate variance of predictions
                    predictions_variance.append(tf.math.reduce_variance(predicted_values, axis=0))

                except Exception as error:
                    print(f'Error processing {particle} at dropout {dropout}: {error}')

        return predictions_variance

    def evaluate_all_models(self):
        prediction_list = []
        for name, model in self.models.items():
            name_split = name.split('_')
            particle = name_split[0]
            dropout = name_split[1]
            prediction_list.append(
                self.evaluate_model(model, particle, dropout)
            )
        return prediction_list




