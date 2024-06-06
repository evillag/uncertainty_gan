import unittest

import numpy as np
import tensorflow as tf

# Example usage:
# Define dummy test_bench and datasets for demonstration purposes
from MCDEvaluator import MCDEvaluator


class DummyModel:
    def get_generator(self):
        return lambda x: tf.constant([[0.5]] * len(x), dtype=tf.float32)
    def ensemble_inference_mode(self):
        pass

models = {
    'muon_0.1': DummyModel(),
    'electron_0.2': DummyModel(),
}

datasets = {
    'muon': {'feats_val': np.random.rand(1000, 10), 'targets_val': np.random.rand(1000)},
    'electron': {'feats_val': np.random.rand(1000, 10), 'targets_val': np.random.rand(1000)},
}

num_reps = 5
sub_sample_size = 100

evaluator = MCDEvaluator(models, datasets, num_reps, sub_sample_size)
results = evaluator.evaluate_all_models()
print(results)


# Tests
class TestMCDEvaluator(unittest.TestCase):

    def setUp(self):
        self.models = {
            'dummy_0.1': DummyModel(),
            'dummy_0.2': DummyModel(),
        }
        self.datasets = {
            'dummy': {'feats_val': np.random.rand(100, 10), 'targets_val': np.random.rand(100)},
        }
        self.num_reps = 3
        self.sub_sample_size = 10
        self.evaluator = MCDEvaluator(self.models, self.datasets, self.num_reps, self.sub_sample_size)

    def test_subsample_dataset(self):
        features = np.array([[i] for i in range(100)])
        targets = np.array([i for i in range(100)])
        subsampled_features, subsampled_targets = self.evaluator.subsample_dataset(features, targets, 10)
        self.assertEqual(len(subsampled_features), 10)
        self.assertEqual(len(subsampled_targets), 10)

    def test_evaluate_model(self):
        dummy_model = DummyModel()
        variance = evaluate_model(dummy_model, 'dummy', 0.1)
        self.assertGreater(len(variance), 0)

    def test_evaluate_all_models(self):
        results = self.evaluator.evaluate_all_models()
        self.assertEqual(len(results), 2)

    def test_inference_predictions_structure(self):
        dummy_model = DummyModel()
        variance = evaluate_model(dummy_model, 'dummy', 0.1)
        self.assertIsInstance(variance, list)

    def test_prediction_variance_calculation(self):
        dummy_model = DummyModel()
        variance = evaluate_model(dummy_model, 'dummy', 0.1)
        self.assertTrue(all(tf.reduce_mean(v).numpy() >= 0 for v in variance))


# Run tests
if __name__ == '__main__':
    unittest.main()
