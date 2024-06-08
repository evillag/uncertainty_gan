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

    # An ensemble must be of at least two models in order to calculate
    # the variance of the predictions
    if ensemble_size == 1:
        ensemble_size += 1

    generator = model.get_generator()
    generator.ensemble_inference_mode()

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
    def __init__(self, model: MonteCarloDropoutModel, ensemble_size=10):
        self.model = model
        self.ensemble_size = ensemble_size

    def evaluate(self, x_sample):
        return evaluate_model(self.model, x_sample, self.ensemble_size)


def test(debug=False):
    pass
    # if debug:
    #     checkpoint_dp = '0.01'
    #     checkpoint_base = '../test_bench/checkpoints/'
    #     dropout_type = 'bernoulli_structured'
    #     data_dir = '../test_bench/rich/'
    #     particles = ['pion', 'muon']
    #     dropouts = [0.05, 0.10]
    #     sub_sample_size = .3
    #
    #     # If this cell is run more than once, previous test_bench are garbage collected and a Checkpoint warning is
    #     # displayed, disregard it.
    #     models = dict()
    #     datasets = {particle: load_particle_datasets(particle, data_dir) for particle in particles}
    #
    #     for particle in particles:
    #         for dp in dropouts:
    #             # Test model creation with debug mode on
    #             models[f"{particle}_{dp}"] = MonteCarloDropoutModel(
    #                 particle,
    #                 dropout_rate=dp,
    #                 checkpoint_dir=checkpoint_base + get_checkpoint_name(particle, checkpoint_dp, dropout_type),
    #                 debug=True
    #             )
    #         mcd_evaluator = MCDEvaluator(model, )
    #         prediction_list = mcd_evaluator.evaluate_all_models()
    #
    #         for i, pred in enumerate(prediction_list):
    #             print('{i} - Predictions Variance and Values:')
    #             print(f'Variance:\n{pred[0]}')
    #             print(f'Value:\n{pred[1]}')


# Run tests
if __name__ == '__main__':
    test(True)
