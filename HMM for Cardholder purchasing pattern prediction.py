import numpy as np

# Define initial state probabilities
initial_state_probs = np.array([0.1148, 0.292, 0.4, 0.5532])

# Transition probability matrix
transition_matrix = np.array([[0.1, 0.3, 0.5, 0.1],
                             [0.2, 0.3, 0.1, 0.4],
                             [0.6, 0.1, 0.1, 0.2],
                             [0.1, 0.3, 0.3, 0.3]])


# Emission probability matrix (observation probabilities)
emission_matrix = np.array([
    [0.2, 0.3, 0.4, 0.1],
    [0.4, 0.2, 0.1, 0.3],
    [0.3, 0.3, 0.2, 0.2],
    [0.1, 0.2, 0.3, 0.4]
])

# Generate a sequence of observations
observations_sequence = np.array([0, 1, 2, 3, 2, 1])

# Expectation-Maximization (EM) algorithm
def expectation_maximization(initial_state_probs, transition_matrix, emission_matrix, observations_sequence, n_iter=100):
    n_states = initial_state_probs.shape[0]
    n_obs = len(observations_sequence)

    # Initialize parameters randomly
    pi = initial_state_probs
    A = transition_matrix
    B = emission_matrix

    for _ in range(n_iter):
        # Forward pass
        alpha = np.zeros((n_obs, n_states))
        alpha[0] = pi * B[:, observations_sequence[0]]
        for t in range(1, n_obs):
            alpha[t] = np.sum(alpha[t - 1].reshape(-1, 1) * A.T * B[:, observations_sequence[t]], axis=0)

        # Backward pass
        beta = np.zeros((n_obs, n_states))
        beta[-1] = 1
        for t in range(n_obs - 2, -1, -1):
            beta[t] = np.sum(A * B[:, observations_sequence[t + 1]] * beta[t + 1], axis=1)

        # Compute gamma and xi
        gamma = alpha * beta / np.sum(alpha * beta, axis=1).reshape(-1, 1)
        xi = np.zeros((n_obs - 1, n_states, n_states))
        for t in range(n_obs - 1):
            xi[t] = (alpha[t].reshape(-1, 1) * A * B[:, observations_sequence[t + 1]] * beta[t + 1]) / \
                    np.sum(alpha[t] @ A * B[:, observations_sequence[t + 1]] * beta[t + 1])

        # Update parameters
        pi = gamma[0]
        for i in range(n_states):
            for j in range(n_states):
                A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
        for j in range(n_states):
            for k in range(emission_matrix.shape[1]):
                B[j, k] = np.sum(gamma[observations_sequence == k, j]) / np.sum(gamma[:, j])

    return pi, A, B

# Run EM algorithm
pi_optimal, A_optimal, B_optimal = expectation_maximization(initial_state_probs, transition_matrix, emission_matrix, observations_sequence)

# Print optimal transition and emission matrices
print("Optimal Transition Matrix after EM:")
print(A_optimal)
print("\nOptimal Emission Matrix after EM:")
print(B_optimal)
