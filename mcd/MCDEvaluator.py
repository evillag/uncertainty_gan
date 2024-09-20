from enum import Enum

import tensorflow as tf
from tqdm import tqdm, trange

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


def evaluate_model(model, x_sample, ensemble_size=10):
    prediction_list = []
    if ensemble_size == 1:
        ensemble_size += 1

    generator = model.get_generator()

    if mode == 'ensemble_inference':
      generator.ensemble_inference_mode()
    else:
      generator.single_model_inference_mode()

    print(f"Generating ensemble({ensemble_size}) predictions")
    for _ in trange(ensemble_size):
        prediction_list.append(generator(x_sample))

    predicted_values = tf.convert_to_tensor(prediction_list)

    # Calculate variance of predictions
    ensemble_variance = tf.math.reduce_variance(predicted_values, axis=0)

    # Predictions mean
    ensemble_mean = tf.math.reduce_mean(predicted_values, axis=0)

    return ensemble_variance, ensemble_mean


class MCDEvaluator:
    def __init__(self, models, datasets, sub_sample_size, ensemble_size=10):
        self.models = models
        self.datasets = datasets
        self.sub_sample_size = sub_sample_size
        self.ensemble_size = ensemble_size

    def evaluate_all_models(self):
        # Draw a sample of the datasets
        x_sample, y_sample = subsample_dataset(self.datasets['feats_val'],
                                               self.datasets['targets_val'],
                                               self.sub_sample_size)
        prediction_list = []
        for name, model in self.models.items():
            name_split = name.split('_')
            particle = name_split[0]
            dropout = name_split[1]
            print(f"Evaluating Model for {particle} with dropout rate of {dropout}")
            prediction_list.append(
                evaluate_model(model, x_sample, self.ensemble_size)
            )
        return prediction_list


def test(debug=False):
    if debug:
        checkpoint_dp = '0.01'
        checkpoint_base = '../test_bench/checkpoints/'
        dropout_type = 'bernoulli_structured'
        data_dir = '../test_bench/rich/'
        particles = ['pion', 'muon']
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
            mcd_evaluator = MCDEvaluator(models, datasets[particle], sub_sample_size)
            prediction_list = mcd_evaluator.evaluate_all_models()

            for i, pred in enumerate(prediction_list):
                print('{i} - Predictions Variance and Values:')
                print(f'Variance:\n{pred[0]}')
                print(f'Value:\n{pred[1]}')


# Run tests
if __name__ == '__main__':
    test(True)
