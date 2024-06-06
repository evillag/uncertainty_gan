import unittest

from test_bench.model import MonteCarloDropoutModel


class TestMonteCarloDropoutModel(unittest.TestCase):

    def setUp(self):
        self.model = MonteCarloDropoutModel('muon', 0.1, debug=False)

    def test_model_initialization(self):
        self.assertEqual(self.model.particle, 'muon')
        self.assertEqual(self.model.dropout_rate, 0.1)
        self.assertIsNotNone(self.model.get_generator())

    def test_checkpoint_name(self):
        self.assertEqual(self.model._get_checkpoint_name(),
                         'checkpoint_dir')

    def test_model_string_representation(self):
        self.assertEqual(str(self.model), 'muon_0.1')

    def test_restore_model(self):
        try:
            self.model._restore_model()
        except Exception as e:
            self.fail(f"Model restore failed: {e}")


if __name__ == '__main__':
    unittest.main()
