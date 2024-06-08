import numpy as np
import tensorflow as tf
from keras import Model

from scipy.stats import gaussian_kde
from tqdm import trange

from test_bench.model import MonteCarloDropoutModel

# Constants
EMBEDDING_LAYER = 14  # Index is 14 because layer indexing starts from 0


def fit_kde(embeddings):
    kde_fit_functions = []
    for feature in range(embeddings.shape[1]):
        kde_fit_functions.append(gaussian_kde(embeddings[:, feature], bw_method=None))
    return kde_fit_functions


def integration_likelihood(embeddings, kde_fit_functions, epsilon=0.01):
    """ This function calculates the likelihood of a given embedding point in the fitted KDE functions by estimating the
    integral between each data point - epsilon and the point + epsilon.

    Arguments expected:
        - embeddings: np.array: embeddings to be evaluated
        - kde_fit_functions: list: list of gaussian_kde functions fitted to the embeddings
        - epsilon: float: epsilon value to be used in the integration
    Returns:
        - likelihoods: tensorflow.Tensor: likelihoods of the points based on the KDE probability density functions.
    """
    n_samples = embeddings.shape[0]
    n_features = embeddings.shape[1]

    likelihoods = np.zeros((n_samples, n_features))
    for i in trange(n_samples):
        for j in range(n_features):
            point = embeddings[i, j]
            likelihoods[i, j] = kde_fit_functions[j].integrate_box_1d(point - epsilon, point + epsilon)
    return tf.convert_to_tensor(likelihoods)


def tf_integration_likelihood(embeddings, kde_fit_functions, epsilon=0.01):
    """ This function calculates the likelihood of a given embedding point in the fitted KDE functions by estimating the
    integral between each data point - epsilon and the point + epsilon.

    Arguments expected:
        - embeddings: np.array: embeddings to be evaluated
        - kde_fit_functions: list: list of gaussian_kde functions fitted to the embeddings
        - epsilon: float: epsilon value to be used in the integration
    Returns:
        - likelihoods: tensorflow.Tensor: likelihoods of the points based on the KDE probability density functions.
    """
    likelihoods = []
    n_features = embeddings.shape[1]
    tf_embeddings = tf.convert_to_tensor(embeddings)

    print('Calculating likelihoods with integration method')
    for j in trange(n_features):
        def integrate_fn(x):
            return kde_fit_functions[j].integrate_box_1d(x - epsilon, x + epsilon)

        likelihoods.append(tf.map_fn(integrate_fn, tf_embeddings[:, j]))
    return tf.transpose(tf.convert_to_tensor(likelihoods))


# previously know as the range_mapping_likelihood.
def tf_calculate_normalized_likelihoods(embeddings, kde_fit_functions, normalization_datapoints=100):
    """ This function calculates the likelihood of a given embedding point in the fitted KDE functions by estimating the
    value of the KDE function at the point divided by the maximum value of the KDE function for the KDE´s feature.

    Arguments expected:
        - embeddings: np.array: embeddings to be evaluated
        - kde_fit_functions: list: list of gaussian_kde functions fitted to the embeddings
    Returns:
        - likelihoods: tensorflow.Tensor: likelihoods of the points based on the KDE probability density functions.
    """
    likelihoods = []
    n_features = embeddings.shape[1]
    x = tf.linspace(embeddings.min(), embeddings.max(), normalization_datapoints)
    embeddings = tf.convert_to_tensor(embeddings)

    print('Calculating normalized likelihoods')
    for j in trange(n_features):
        def normalize_kde_fn(row_and_kde_max):
            row, fit_max = row_and_kde_max
            return kde_fit_functions[j](row) / fit_max

        kde_max = tf.ones(embeddings.shape[0], dtype=tf.float32) * kde_fit_functions[j](x).max()
        likelihoods.append(tf.map_fn(normalize_kde_fn, elems=(embeddings[:, j], kde_max)))
    return tf.transpose(tf.convert_to_tensor(likelihoods))


# previously know as the range_mapping_likelihood.
def calculate_normalized_likelihoods(embeddings, kde_fit_functions, normalization_datapoints=100):
    """ Numpy version: This function calculates the likelihood of a given embedding point in the fitted KDE functions by
    estimating the value of the KDE function at the point divided by the maximum value of the KDE function for the KDE´s
    feature.

    Arguments expected:
        - embeddings: np.array: embeddings to be evaluated
        - kde_fit_functions: list: list of gaussian_kde functions fitted to the embeddings
    Returns:
        - likelihoods: tensorflow.Tensor: likelihoods of the points based on the KDE probability density functions.
    """
    n_samples = embeddings.shape[0]
    n_features = embeddings.shape[1]

    likelihoods = np.zeros((n_samples, n_features))

    x = np.linspace(embeddings.min(), embeddings.max(), normalization_datapoints)

    for j in trange(n_features):
        kde_max = kde_fit_functions[j](x).max()
        for i in range(n_samples):
            point = embeddings[i, j]
            likelihoods[i, j] = kde_fit_functions[j](point) / kde_max
    return tf.convert_to_tensor(likelihoods)


def create_embeddings_model(original_model: MonteCarloDropoutModel, embedding_layer=EMBEDDING_LAYER) -> Model:
    # 1. Set the model in inference mode
    generator = original_model.get_generator()
    generator.single_model_inference_mode()

    # 2. Create a new model that exposes the layer(s) of interest
    input_layer = generator.input
    output_layer = generator.layers[embedding_layer].output

    # 3. Create a `new_model` without optimizations
    embeddings_model = Model(input_layer, [output_layer, generator.output])
    return embeddings_model


def evaluate_model(model: MonteCarloDropoutModel, x_sample, kde_fit_functions, likelihood_method='integration',
                   embeddings_model: Model = None, use_tf_version=False, normalization_datapoints=100):
    """ Arguments expected:
        - model: keras model: model to be evaluated
        - x_sample: np.array: sample to be evaluated
        - kde_fit_functions: Any: Fitted KDE functions to be used in the evaluation, there is one per dimension of
        x_sample
        - unc_method: str: method to be used to calculate the uncertainty. Options are 'integration' and 'normalized'
    Returns:
        - Tuple(tensorflow.Tensor, np.array): Tuple with uncertainty score (complement of likelihood) for the sample and
         the generated targets.
    """
    integration_fn = integration_likelihood if not use_tf_version else tf_integration_likelihood
    normalized_fn = calculate_normalized_likelihoods if not use_tf_version else tf_calculate_normalized_likelihoods

    if embeddings_model is None:
        print('Generating an embeddings model')
        embeddings_model = create_embeddings_model(model)

    print('Calculating sample´s embeddings')
    sample_embeddings, sample_predictions = embeddings_model.predict(x_sample)

    print('Estimating sample´s feature densities')
    feature_densities = tf.constant(np.ones((x_sample.shape[0], 1)))
    if likelihood_method == 'integration':
        feature_densities = integration_fn(sample_embeddings, kde_fit_functions)
    elif likelihood_method == 'normalized':
        feature_densities = normalized_fn(sample_embeddings, kde_fit_functions, normalization_datapoints)

    reduced_feature_densities = tf.math.reduce_mean(feature_densities, axis=1)
    return 1.0 - reduced_feature_densities, sample_predictions


class FeatureDensityEvaluator:
    def __init__(self, model: MonteCarloDropoutModel, known_embeddings=None, likelihood_method='integration',
                 use_tf_version=False, normalization_datapoints=100):
        self.model = model
        self.known_embeddings = known_embeddings
        self.likelihood_method = likelihood_method
        self.use_tf_version = use_tf_version

        print('Generating an embeddings model')
        self.embeddings_model = create_embeddings_model(model)

        print('Fitting KDE functions to known embeddings')
        self.kde_fit_functions = fit_kde(known_embeddings)

        # used in normalization method
        self.normalization_datapoints = normalization_datapoints

    def evaluate(self, x_sample):
        uncertainties, generated_targets = evaluate_model(
            self.model, x_sample, self.kde_fit_functions, self.likelihood_method, self.embeddings_model,
            self.use_tf_version, self.normalization_datapoints)

        return uncertainties, generated_targets
