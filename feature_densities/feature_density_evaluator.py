import numpy as np
import tensorflow as tf
from tf_keras import Model

from scipy.stats import gaussian_kde
from tqdm import trange
from tqdm import tqdm

from test_bench.model import MonteCarloDropoutModel

# Constants
EMBEDDING_LAYER = 14  # Index is 14 because layer indexing starts from 0


def scott_bandwidth(data, features=128):
    """Calculates the bandwidth for Kernel Density Estimation using Scott's Rule.
    
    Arguments:
        - data: tf.Tensor: Input data points for KDE
        - features: int: Number of features (dimensions) in the data.
    
    Returns:
        - bandwidth: float: Calculated bandwidth based on Scott's Rule.
    """
    # Number of samples
    n_samples = tf.shape(data)[0]
    n_features = features
    
    # Apply Scott's rule
    bandwidth = tf.pow(tf.cast(n_samples, tf.float32), -1 / (4 + tf.cast(n_features, tf.float32)))

    return bandwidth

@tf.function
def gaussian_kernel(x_values, data, bandwidth):
    """ This function computes a Gaussian Kernel Density Estimate (KDE) for given data points, optimized for GPU execution with TensorFlow.
    
    Arguments:
        - x_values: tf.Tensor: points at which to evaluate the KDE.
        - data: tf.Tensor: data points used to fit the KDE.
        - bandwidth: float: bandwidth parameter, which controls the smoothness of the KDE.
    
    Returns:
        - density: tf.Tensor: estimated KDE density values at each point in x_values.
    """
    n = tf.shape(data)[0]

    diff = tf.expand_dims(x_values, axis=1) - data

    density = tf.exp(-0.5 * (diff / bandwidth) ** 2) / (bandwidth * tf.sqrt(tf.constant(2 * np.pi, dtype=tf.float32)))

    return tf.reduce_sum(density, axis=1) / tf.cast(n, tf.float32)

def get_kde_max(known_embeddings, n_samples=100, kde_max_resolution=10000, n_features=128):
    """This function calculates the maximum value of the KDE function for the KDE's
    feature used in the normalized likelihood function

    Arguments expected:
        - known_embeddings: np.array: given embeddings to calculate the kde function.
        - n_samples: int: number of samples to use.
        - kde_max_resolution: int: number of points computed to normalize output likelihoods.
        - n_features: int: number of features to calculate.
    Returns:
        - kde_max: np.array: list of values of the maximum value of the KDE function for the KDE's
          feature.
    """

    known_embeddingsT = known_embeddings.T
    known_embeddings_tf = tf.convert_to_tensor(known_embeddingsT, dtype=tf.float32)


    @tf.function
    def mapFunction(j):
       known_feature = tf.gather(known_embeddings_tf, j)
       x = tf.linspace(tf.reduce_min(known_feature), tf.reduce_max(known_feature), kde_max_resolution)
       data_slice = tf.gather(known_embeddings[:n_samples], j, axis=1)
       
       return tf.reduce_max(gaussian_kernel(x, data_slice, scott_bandwidth(data_slice)))

    kde_max = tf.map_fn(mapFunction, tf.range(n_features), fn_output_signature=tf.float32)


    return kde_max

def fit_kde(embeddings, n_samples=100):
    """ This function calculates the likelihood of a given embedding point in the fitted KDE functions by estimating the
    integral between each data point - epsilon and the point + epsilon.

    Arguments expected:
        - embeddings: np.array: embeddings to fit.
        - n_samples: int | None: number of samples used for KDE fitting. If None, uses all samples from embeddings.
    Returns:
        - kde_fit_functions: list: list of gaussian_kde functions fitted to the embeddings.
    """
    kde_fit_functions = []

    for feature in range(embeddings.shape[1]):
        kde_fit_functions.append(gaussian_kde(embeddings[:n_samples, feature]))

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
def tf_calculate_normalized_likelihoods(embeddings, kde_fit_functions):
    """ This function calculates the likelihood of a given embedding point in the fitted KDE functions by estimating the
    value of the KDE function at the point divided by the maximum value of the KDE function for the KDE's feature.

    Arguments expected:
        - embeddings: np.array: embeddings to be evaluated
        - kde_fit_functions: list: list of gaussian_kde functions fitted to the embeddings
    Returns:
        - likelihoods: tensorflow.Tensor: likelihoods of the points based on the KDE probability density functions.
    """
    likelihoods = []
    n_features = embeddings.shape[1]

    x = np.linspace(embeddings.min(), embeddings.max(), 10000)
    kde_max_values = [kde_fn(x).max() for kde_fn in kde_fit_functions]
    tf_embeddings = tf.convert_to_tensor(embeddings)

    print('Calculating normalized likelihoods')

    for j in trange(n_features):
        def normalize_kde_fn(x):
            return kde_fit_functions[j](x) / kde_max_values[j]

        likelihoods.append(tf.map_fn(normalize_kde_fn, tf_embeddings[:, j]))

    return tf.transpose(tf.convert_to_tensor(likelihoods))


# previously know as the range_mapping_likelihood.
def calculate_normalized_likelihoods(known_embeddings, embeddings, kde_max, n_samples=100):
    """ Numpy version: This function calculates the likelihood of a given embedding point in the fitted KDE functions by
    estimating the value of the KDE function at the point divided by the maximum value of the KDE function for the KDE's
    feature.

    Arguments expected:
        - embeddings: np.array: embeddings to be evaluated.
        - kde_fit_functions: list: list of gaussian_kde functions fitted to the embeddings.
        - kde_max_resolution: int: number of points computed to normalize output likelihoods.
    Returns:
        - likelihoods: tensorflow.Tensor: likelihoods of the points based on the KDE probability density functions.
    """
  
    embeddingsT = tf.transpose(tf.convert_to_tensor(embeddings, dtype=tf.float32))
    known_embeddings = tf.convert_to_tensor(known_embeddings[:n_samples], dtype=tf.float32)
    kde_max = tf.convert_to_tensor(kde_max, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(tf.range(tf.shape(embeddingsT)[0]))

    @tf.function
    def myFunction(j):
        result = gaussian_kernel(embeddingsT[j], known_embeddings[:, j], scott_bandwidth(known_embeddings[:, j])) / kde_max[j]
        return result

    # Aplicar la función usando map, que se ejecutará en GPU
    mapped_results = tqdm(dataset.map(lambda j: myFunction(j), num_parallel_calls=tf.data.AUTOTUNE), total=embeddings.shape[1])

    # Convertir resultados en un tensor
    mapped_tensor = tf.stack(list(mapped_results))
    return tf.transpose(mapped_tensor)


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

def  generate_kde_fit_functions(known_embeddings, n_fit_samples=None):
    """ This function fits KDE functions to known embeddings.
    Arguments expected:
        - known_embeddings: np.array: embeddings already known to the model, likely the embeddings from the training set
        - n_fit_samples: int | None: number of samples used for KDE fitting. If None, uses all samples from embeddings.
    """
    print('Fitting KDE functions to known embeddings')
    kde_fit_functions = fit_kde(known_embeddings, n_fit_samples)
    return kde_fit_functions


def evaluate_model(model: MonteCarloDropoutModel, x_sample, kde_max, likelihood_method='integration',
                   known_embeddings=None, embedding_layer=EMBEDDING_LAYER, n_samples=100):
    """ This function evaluates a sample using the features densities uncertainty method.
    Arguments expected:
        - model: keras model: model to be evaluated.
        - x_sample: np.array: sample to be evaluated.
        - kde_fit_functions: list: list of gaussian_kde functions fitted to the embeddings.
        - likelihood_method: str: method to be used to calculate the uncertainty. Options are 'integration' and
          'normalized'.
        - known_embeddings: np.array: embeddings already known to the model, likely the embeddings from the training set.
          Required if the method is 'normalized'.
        - embedding_layer: int: index of the layer to be used as embeddings.
    Returns:
        - Tuple(tensorflow.Tensor, np.array): Tuple with uncertainty score (complement of likelihood) for the sample and
          the generated targets.
    """
    print('Generating an embeddings model')
    embeddings_model = create_embeddings_model(model, embedding_layer)

    print('Calculating sample\'s embeddings')
    sample_embeddings, sample_predictions = embeddings_model(x_sample)

    print('Estimating sample\'s feature densities')
    feature_densities = tf.constant(np.ones((x_sample.shape[0], 1)))
    if likelihood_method == 'integration':
        feature_densities = integration_likelihood(sample_embeddings, generate_kde_fit_functions(known_embeddings, n_samples))
    elif likelihood_method == 'normalized':
        feature_densities = calculate_normalized_likelihoods(known_embeddings, sample_embeddings, kde_max, n_samples=n_samples)

    reduced_feature_densities = tf.math.reduce_mean(feature_densities, axis=1)
    return 1.0 - reduced_feature_densities, sample_predictions
