from enum import Enum

import tensorflow as tf
from tqdm import tqdm

from test_bench import get_checkpoint_name, load_particle_datasets, subsample_dataset
from test_bench.model import MonteCarloDropoutModel


class RichDLL(Enum):
    RichDLLe = 1
    RichDLLk = 2
    RichDLLmu = 3
    RichDLLp = 4
    RichDLLbt = 5

    @staticmethod
    def get_by_alternate_name(name):
        mapping = {
            "pion": RichDLL.RichDLLe,
            "kaon": RichDLL.RichDLLk,
            "muon": RichDLL.RichDLLmu,
            "proton": RichDLL.RichDLLp,
            "tresshold": RichDLL.RichDLLbt
        }
        return mapping.get(name, None)


class MCDEvaluator:
    def __init__(self, models, datasets, num_reps, sub_sample_size):
        self.models = models
        self.datasets = datasets
        self.num_reps = num_reps
        self.sub_sample_size = sub_sample_size

    def evaluate_model(self, model, particle, dropout):
        generator = model.get_generator()
        generator.ensemble_inference_mode()
        ensemble_inferences = dict()
        predictions_variance = []
        predictions_mean = []

        print(f"Generating predictions with model for particle {particle.upper()} and dropout {dropout}")
        for ensemble_size in tqdm(range(1, 11)):  # Assuming ensemble sizes ranges from 1 to 10
            inference_predictions = dict()
            for rep in range(self.num_reps):
                try:
                    key = f'{rep}'
                    ensemble_inferences[f'{ensemble_size}'] = {key: inference_predictions}

                    # Draw a sample of the datasets
                    x_sample, y_sample = subsample_dataset(self.datasets[particle]['feats_val'],
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

                    # Calculate variance of predictions
                    predictions_variance.append(tf.math.reduce_variance(predicted_values, axis=0))

                    # Predictions mean
                    predictions_mean.append(tf.math.reduce_mean(predicted_values, axis=0))

                except Exception as error:
                    print(f'Error processing {particle} at dropout {dropout}: {error}')

        return predictions_variance, predictions_mean

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


def test(debug=False):
    if debug:
        checkpoint_dp = '0.01'
        checkpoint_base = '../../checkpoints/'
        dropout_type = 'bernoulli_structured'
        data_dir = '../../data/rich/'
        num_reps = 10
        particles = ["muon"]
        dropouts = [0.05, 0.10]
        sub_sample_size = .3

        # If this cell is run more than once, previous test_bench are garbage collected and a Checkpoint warning is
        # displayed, disregard it.
        models = dict()
        datasets = {particle: load_particle_datasets(particle, data_dir) for particle in particles}
        for particle in particles:
            for dp in dropouts:
                # Test model creation with debug mode on
                models[f"{particle}_{dp}"] = MonteCarloDropoutModel(
                    particle,
                    dropout_rate=dp,
                    checkpoint_dir=checkpoint_base + get_checkpoint_name(particle, checkpoint_dp, dropout_type),
                    debug=True
                )
            mcdEvaluator = MCDEvaluator(models, datasets, num_reps, sub_sample_size)
            prediction_list = mcdEvaluator.evaluate_all_models()

            for i, pred in enumerate(prediction_list):
                print('{i} - Predictions Variance and Values:')
                print(f'Variance:\n{pred[0]}')
                print(f'Value:\n{pred[1]}')


# Run tests
if __name__ == '__main__':
    test(True)
