import tensorflow as tf
import os


from src.cramer_gan_trainer import CramerGANTrainer
from src.dataset import CramerGANDataset
from src.datasets.utils_rich import (get_merged_typed_dataset,
                                     parse_dataset_np, parse_example)
from src.models.gans.discriminators.fcn_disc import RICHDiscriminator
from src.models.gans.generators.fcn_gen import RichMCDropFunc
import tf_keras
from tf_keras.optimizers.legacy import RMSprop

class MonteCarloDropoutModel:

    def __init__(self, particle, dropout_rate, checkpoint_base='CHECKPOINT_BASE',
                 checkpoint_file='CKPT_NUMBER', debug=False):
        self.particle = particle
        self.dropout_rate = dropout_rate

        print(f'Generating model for {particle} with a dropout rate of {dropout_rate}')

        self._gen_config = {
            'drop_rate': dropout_rate,
            'dropout_type': 'bernoulli',
        }

        self._generator = self._build_generator()
        self._discriminator = self._build_discriminator()
        self._checkpoint_dir = os.path.join(checkpoint_base, self._get_checkpoint_name())
        self._filename = os.path.join(self._checkpoint_dir, checkpoint_file)

        print("----",self._filename)

        if debug:
            self._print_model_summaries()

        self._generator_optimizer = RMSprop(2e-4)
        self._discriminator_optimizer = RMSprop(2e-4)

        self._trainer_config = {
            'generator': self._generator,
            'discriminator': self._discriminator,
            'generator_optimizer': self._generator_optimizer,
            'discriminator_optimizer': self._discriminator_optimizer,
            'checkpoint_dir': self._checkpoint_dir
        }

        self.trainer = self._initialize_trainer()
        self._restore_model()

    def _get_checkpoint_name(self):
        return f'bernoulli_structured_dropout_line_test_cramer_weighted_{self.particle}'

    def __str__(self):
        return f"{self.particle}_{self.dropout_rate}"

    def get_generator(self):
        return self._generator

    def _build_generator(self):
        generator = RichMCDropFunc(**self._gen_config)
        generator.build((None, 3))
        return generator

    def _build_discriminator(self):
        return RICHDiscriminator()

    def _print_model_summaries(self):
        print("\nGenerator:\n")
        print(self._generator.summary(line_length=96))
        print("\nDiscriminator:\n")
        print(self._discriminator.summary())
        print(f"\nCheckpoint filename: {self._filename}\n")

    def _initialize_trainer(self):
        return CramerGANTrainer(**self._trainer_config)

    def _restore_model(self):
        self.trainer.restore(self._filename)


def test(debug=False):
    if debug:
        DATA_DIR = '/content/drive/MyDrive/data/rich'
        CHECKPOINT_BASE = 'checkpoints/'
        CKPT_NUMBER = 'ckpt-21'

        # Test model creation with debug mode on
        model = MonteCarloDropoutModel(
            'muon',
            dropout_rate=0.5,
            checkpoint_base=CHECKPOINT_BASE,
            checkpoint_file=CKPT_NUMBER,
            debug=True
        )
        print(model)

# Run tests
if __name__ == '__main__':
    test(False)
