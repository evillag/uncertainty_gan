from tf_keras.optimizers.legacy import RMSprop

from src.cramer_gan_trainer import CramerGANTrainer
from src.models.gans.discriminators.fcn_disc import RICHDiscriminator
from src.models.gans.generators.fcn_gen import RichMCDropFunc, VirtualEnsembleModel


class MonteCarloDropoutModel:
    def __init__(self, particle, dropout_rate,
                 log_dir='log_dir_tmp',
                 checkpoint_dir='checkpoint_dir',
                 dropout_type='bernoulli_structured',
                 debug=False):

        self.particle = particle
        self.dropout_rate = dropout_rate
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        print(f'Generating model for {particle} with a dropout rate of {dropout_rate}')

        self._gen_config = {
            'drop_rate': dropout_rate,
            'dropout_type': dropout_type,
        }

        self._generator = RichMCDropFunc(**self._gen_config)
        self._generator.build((None, 3))
        self._discriminator = RICHDiscriminator()

        if debug:
            print("\nGenerator:\n")
            print(self._generator.summary(line_length=96))
            print("\nDiscriminator:\n")
            print(self._discriminator.summary())
            print(f"\nCheckpoint path: {self.checkpoint_dir}\n")

        # Model was trained with tensorflow 2.10.1, use the legacy optimizer
        self._generator_optimizer = RMSprop(2e-4)
        self._discriminator_optimizer = RMSprop(2e-4)

        self._trainer_config = {
            'generator': self._generator,
            'discriminator': self._discriminator,
            'generator_optimizer': self._generator_optimizer,
            'discriminator_optimizer': self._discriminator_optimizer,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': log_dir
        }

        self._restore_model()

    def _get_checkpoint_name(self):
        return self.checkpoint_dir.split('/')[-1]

    def _restore_model(self):
        trainer = CramerGANTrainer(**self._trainer_config)
        trainer.restore_last()

    def __str__(self):
        return f"{self.particle}_{self.dropout_rate}"

    def get_generator(self) -> VirtualEnsembleModel:
        return self._generator
