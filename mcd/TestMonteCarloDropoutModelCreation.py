from test_bench.model import MonteCarloDropoutModel


def test(debug=False):
    if debug:
        checkpoint_base = '../checkpoints/'

        # Test model creation with debug mode on
        model = MonteCarloDropoutModel(
            'muon',
            dropout_rate=0.5,
            checkpoint_dir=checkpoint_base,
            debug=True
        )
        print(model)


# Run tests
if __name__ == '__main__':
    test(True)
