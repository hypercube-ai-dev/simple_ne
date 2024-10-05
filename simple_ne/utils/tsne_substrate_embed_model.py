import torch
import torch.nn.functional as F

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def _compute_pairwise_distances(self, X):
        """
        Compute pairwise Euclidean distances in high-dimensional space.
        """
        sum_X = torch.sum(X ** 2, dim=1).view(-1, 1)
        distances = sum_X + sum_X.t() - 2 * torch.mm(X, X.t())
        return distances

    def _compute_joint_probabilities(self, distances, perplexity):
        """
        Compute conditional probabilities P_ij in high-dimensional space using a Gaussian kernel.
        """
        beta = torch.ones(distances.shape[0], 1)  # Inverse variance of the Gaussian
        P = torch.zeros_like(distances)
        for i in range(distances.shape[0]):
            betamin = -torch.inf
            betamax = torch.inf
            # Perform binary search to get the right beta that matches perplexity
            for _ in range(50):
                P_i = torch.exp(-distances[i] * beta[i])
                P_i[i] = 0  # Remove self-similarity
                sum_P_i = torch.sum(P_i)
                H_i = torch.log(sum_P_i) + beta[i] * torch.sum(distances[i] * P_i) / sum_P_i
                P[i] = P_i / sum_P_i
                H_diff = H_i - torch.log(torch.tensor(perplexity))
                if torch.abs(H_diff) < 1e-5:
                    break
                if H_diff > 0:
                    betamin = beta[i].clone()
                    if betamax == torch.inf:
                        beta[i] *= 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i].clone()
                    if betamin == -torch.inf:
                        beta[i] /= 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2
        P = (P + P.t()) / (2 * P.shape[0])  # Symmetrize and normalize
        return P

    def _student_t_distribution(self, Y):
        """
        Compute pairwise affinities Q_ij in low-dimensional space using a Student's t-distribution.
        """
        sum_Y = torch.sum(Y ** 2, dim=1).view(-1, 1)
        distances_Y = sum_Y + sum_Y.t() - 2 * torch.mm(Y, Y.t())
        Q = 1 / (1 + distances_Y)
        Q.fill_diagonal_(0)
        Q /= torch.sum(Q)
        return Q

    def fit_transform(self, X):
        """
        Fit the model to the high-dimensional data X and transform it to low-dimensional space.
        """
        X = torch.tensor(X, dtype=torch.float32)
        n_samples = X.shape[0]

        # 1. Compute pairwise distances in high-dimensional space
        distances = self._compute_pairwise_distances(X)

        # 2. Compute joint probabilities P_ij in high-dimensional space
        P = self._compute_joint_probabilities(distances, self.perplexity)

        # 3. Initialize Y (low-dimensional embeddings)
        Y = torch.randn(n_samples, self.n_components, requires_grad=True)

        optimizer = torch.optim.Adam([Y], lr=self.learning_rate)

        # 4. Optimize Y to minimize KL divergence
        for _ in range(self.n_iter):
            optimizer.zero_grad()

            # 5. Compute pairwise affinities Q_ij in low-dimensional space
            Q = self._student_t_distribution(Y)

            # 6. Compute Kullback-Leibler divergence loss (cost function)
            loss = torch.sum(P * torch.log((P + 1e-9) / (Q + 1e-9)))

            # 7. Backpropagation and gradient descent
            loss.backward()
            optimizer.step()

        return Y.detach().numpy()
