"""Multinomial logit fitting shared by M2 and later models.

features has one row per option for each visit: shape (visits, options, weights).
An option's score is its row times the weights; probabilities are the softmax of the
scores. Weights maximise the likelihood of the chosen options minus
penalty / 2 x sum of squared weights, over the weights marked in `penalised`.
Newton's method with step halving; raises if the gradient does not reach TOLERANCE.
"""

import numpy as np


TOLERANCE = 1e-8
MAX_STEPS = 100


def softmax(scores):
    shifted = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def objective(weights, features, chosen, penalty, penalised):
    scores = features @ weights
    peak = scores.max(axis=1, keepdims=True)
    log_totals = (peak + np.log(np.exp(scores - peak).sum(axis=1, keepdims=True)))[:, 0]
    loss = -(scores[np.arange(len(chosen)), chosen] - log_totals).sum()
    return loss + penalty / 2 * (penalised * weights ** 2).sum()


def gradient_and_hessian(weights, features, chosen, penalty, penalised):
    probabilities = softmax(features @ weights)
    expected = np.einsum("nj,njk->nk", probabilities, features)
    gradient = -(features[np.arange(len(chosen)), chosen] - expected).sum(axis=0) + penalty * penalised * weights
    hessian = (np.einsum("nj,njk,njl->kl", probabilities, features, features)
               - expected.T @ expected + penalty * np.diag(penalised))
    return gradient, hessian


def fit_weights(features, chosen, penalty, penalised):
    weights = np.zeros(features.shape[2])
    for step in range(1, MAX_STEPS + 1):
        gradient, hessian = gradient_and_hessian(weights, features, chosen, penalty, penalised)
        if np.abs(gradient).max() < TOLERANCE:
            return weights, step - 1
        direction = np.linalg.solve(hessian, gradient)
        size, current = 1.0, objective(weights, features, chosen, penalty, penalised)
        while objective(weights - size * direction, features, chosen, penalty, penalised) > current and size > 1e-10:
            size /= 2
        weights = weights - size * direction
    raise RuntimeError(f"Newton's method did not converge in {MAX_STEPS} steps.")
