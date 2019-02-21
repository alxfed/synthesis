import numpy as np


def create_regression_dataset(numberof_points=100,
                              numberof_variables=100,
                              model_rank=10,
                              numberof_hidden_variables=1,
                              model_bias=0.0,
                              noise=0.0,
                              coef=False,
                              random_seed=123):
    """
    Builds an underlying model based on model_rank basis of random vectors,
    then applyes it to an array of size numberof_points*numberof_variables
    of random Gaussian distributed points around 0 (plus and minus values).
    :param model_rank:
    :param numberof_hidden_variables:
    :param model_bias: the "b" in w*X + b
    :param noise:
    :param coef:
    :param random_seed:
    :return:
    """
    model_rank = min(numberof_variables, model_rank)
    generator = np.random.RandomState(random_seed)   # an instance of RandomState
    underlying_model = np.zeros((numberof_variables, numberof_hidden_variables))
    underlying_model[:model_rank, :] = 100 * generator.rand(
        model_rank, numberof_hidden_variables)

    X = generator.randn(numberof_points, numberof_variables)
    y = np.dot(X, underlying_model) + model_bias # X assignment?

    # Add noise
    if noise > 0.0:
        y += generator.normal(scale=noise, size=y.shape)

    y = np.squeeze(y)

    if coef:
        return X, y, np.squeeze(underlying_model)

    else:
        return X, y

Xa, ya, coefa = create_regression_dataset(
                numberof_points=4,
                numberof_variables=2,
                model_rank=10,
                numberof_hidden_variables=2,
                model_bias=0.0,
                noise=0.0,
                coef=True,
                random_seed=123)

print(Xa, "\n")
print(ya, "\n")
print(coefa, "\n")

