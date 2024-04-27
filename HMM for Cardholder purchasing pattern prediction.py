import numpy as np

# Dataset: Sector and Number of transactions
data = {
    'Service Centers': 287,
    'Social Joints': 730,
    'Health': 100,
    'Restaurants': 1383
}

# Initial probabilities (random initialization)
initial_probs = np.random.rand(len(data))
initial_probs /= initial_probs.sum()

# Emission probabilities (random initialization)
emission_probs = np.random.rand(len(data), len(data))
emission_probs /= emission_probs.sum(axis=1, keepdims=True)

# Observations (based on sectors)
observations = list(data.values())

# Number of states
num_states = len(data)

# Number of observations
num_observations = len(observations)


def forward_algorithm(obs_seq, init_probs, emiss_probs):
    """
    Forward Algorithm: Computes the forward probabilities for a given observation sequence.
    """
    alpha = np.zeros((len(obs_seq), len(init_probs)))
    # Initialization
    alpha[0] = init_probs * emiss_probs[:, 0]

    # Induction
    for t in range(1, len(obs_seq)):
        for j in range(len(init_probs)):
            alpha[t, j] = np.sum(alpha[t - 1] * emiss_probs[:, j]) * obs_seq[t]

    return alpha


def backward_algorithm(obs_seq, emiss_probs):
    """
    Backward Algorithm: Computes the backward probabilities for a given observation sequence.
    """
    beta = np.zeros((len(obs_seq), len(emiss_probs)))

    # Initialization
    beta[-1] = 1

    # Induction
    for t in range(len(obs_seq) - 2, -1, -1):
        for i in range(len(emiss_probs)):
            beta[t, i] = np.sum(beta[t + 1] * emiss_probs[:, i] * obs_seq[t + 1])

    return beta


def expectation_maximization(observations, initial_probs, emission_probs, num_iterations=100):
    """
    Expectation-Maximization Algorithm: Updates HMM parameters using EM.
    """
    for iteration in range(num_iterations):
        # E-step
        forward = forward_algorithm(observations, initial_probs, emission_probs)
        backward = backward_algorithm(observations, emission_probs)
        xi = np.zeros((len(observations) - 1, len(emission_probs), len(emission_probs)))

        for t in range(len(observations) - 1):
            numerator = np.dot(forward[t, :].reshape(-1, 1), backward[t + 1, :].reshape(1, -1)) * \
                        emission_probs * observations[t + 1]
            denominator = np.sum(numerator)
            xi[t, :, :] = numerator / denominator

        gamma = np.sum(xi, axis=1)

        # M-step
        initial_probs = gamma[0, :]
        emission_probs = np.sum(xi, axis=0) / np.sum(gamma, axis=0).reshape(-1, 1)

    return initial_probs, emission_probs


if __name__ == "__main__":
    initial_probs_est, emission_probs_est = \
        expectation_maximization(observations, initial_probs, emission_probs)

    print("Estimated Initial Probabilities:")
    print(initial_probs_est)
    print("Estimated Emission Probabilities:")
    print(emission_probs_est)

    # Calculate log-likelihood
    forward = forward_algorithm(observations, initial_probs_est, emission_probs_est)
    log_likelihood = np.sum(np.log(forward[-1]))

    print("Log-Likelihood:", log_likelihood)
