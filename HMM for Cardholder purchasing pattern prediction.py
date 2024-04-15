import numpy as np

# Initial state probabilities
initial_prob = np.array([0.4, 0.2, 0.2, 0.2])

# Observation data (Number of transactions)
observations = np.array([443, 216, 141, 1200])

# Initialize transition matrix
transition_matrix = np.array([[0.1, 0.3, 0.5, 0.1],
                              [0.2, 0.3, 0.1, 0.4],
                              [0.6, 0.1, 0.1, 0.2],
                              [0.1, 0.3, 0.3, 0.3]])

# Number of states
num_states = transition_matrix.shape[0]

# Number of observations
num_observations = len(observations)

# Epsilon value for convergence check
epsilon = 1e-6

# Forward algorithm
def forward_algorithm(observations, initial_prob, transition_matrix):
    alpha = np.zeros((len(observations), len(transition_matrix)))
    alpha[0] = initial_prob * observations[0]
    for t in range(1, len(observations)):
        alpha[t] = np.dot(alpha[t - 1], transition_matrix) * observations[t]
    return alpha

# Backward algorithm
def backward_algorithm(observations, transition_matrix):
    beta = np.zeros((len(observations), len(transition_matrix)))
    beta[-1] = 1
    for t in range(len(observations) - 2, -1, -1):
        beta[t] = np.dot(transition_matrix, observations[t + 1] * beta[t + 1])
    return beta

# EM Algorithm
def EM_algorithm(observations, initial_prob, transition_matrix, epsilon=1e-6):
    converged = False
    while not converged:
        # Expectation step
        forward_prob = forward_algorithm(observations, initial_prob, transition_matrix)
        backward_prob = backward_algorithm(observations, transition_matrix)

        # Normalization
        gamma = forward_prob * backward_prob
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        # Maximization step
        new_initial_prob = gamma[0]
        new_transition_matrix = np.sum(forward_prob[:-1, :, None] * transition_matrix[None, :, :] *
                                       backward_prob[1:, None, :] * observations[1:, None, None] /
                                       np.sum(forward_prob[-1]), axis=0)

        # Check for convergence
        if np.max(np.abs(new_transition_matrix - transition_matrix)) < epsilon:
            converged = True

        # Update transition matrix and initial probabilities
        transition_matrix = new_transition_matrix
        initial_prob = new_initial_prob

    return transition_matrix

# Train the model using EM algorithm
optimized_transition_matrix = EM_algorithm(observations, initial_prob, transition_matrix, epsilon)

# Print the optimized transition matrix
print("Optimized Transition Matrix:")
print(optimized_transition_matrix)
