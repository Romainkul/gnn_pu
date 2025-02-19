import numpy as np
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

class GRAB:
    def __init__(self, G, P, U, X, classifier, theta_init):
        self.G = G  # Graph structure, adjacency list or matrix
        self.P = P  # Positive nodes
        self.U = U  # Unlabeled nodes
        self.X = X  # Feature matrix
        self.f = classifier  # Classifier model f with parameters θ
        self.theta = theta_init  # Initial parameters θ
        self.pi_p_est = 0  # Initial class prior estimate

    def loopy_belief_propagation(self, pi_p):
        """Runs loopy belief propagation to estimate beliefs."""
        # Initial potentials
        phi = {i: (pi_p, 1 - pi_p) if i in self.U else (1, 0) for i in self.G.keys()}
        messages = {edge: (0.5, 0.5) for edge in self.G.edges}
        
        converged = False
        while not converged:
            new_messages = {}
            for (i, j) in self.G.edges:
                for v in [+1, -1]:
                    msg = sum(phi[i][u] * self.edge_potential(u, v) * np.prod(
                        [messages[(k, i)][u] for k in self.G.neighbors(i) if k != j])
                              for u in [+1, -1])
                    new_messages[(i, j)] = (msg / sum(msg for msg in messages[(i, j)]),)

            converged = all(np.allclose(new_messages[edge], messages[edge]) for edge in self.G.edges)
            messages = new_messages

        beliefs = {i: [phi[i][u] * np.prod([messages[(k, i)][u] for k in self.G.neighbors(i)])
                       for u in [+1, -1]] for i in self.G.keys()}
        return beliefs

    def edge_potential(self, vi, vj, alpha=0.9):
        """Edge potential function, controls homophily."""
        return alpha if vi == vj else 1 - alpha

    def compute_loss(self, theta, X, y, B, P, U):
        """Objective function for risk minimization."""
        f_preds = self.f(X, theta)
        loss = CrossEntropyLoss()(f_preds[P], y[P])

        for j in U:
            for zj in [+1, -1]:
                loss += -B[j][zj] * torch.log(f_preds[j][zj])
        return loss

    def fit(self):
        optimizer = Adam(self.f.parameters())
        prev_loss = float('inf')
        
        while True:
            # Step 1: Marginalization (belief propagation)
            B = self.loopy_belief_propagation(self.pi_p_est)
            
            # Step 2: Update parameters to minimize the objective function
            optimizer.zero_grad()
            loss = self.compute_loss(self.theta, self.X, self.P, B, self.P, self.U)
            loss.backward()
            optimizer.step()
            
            # Update pi_p estimation
            f_preds = self.f(self.X, self.theta)
            self.pi_p_est = sum(1 for i in self.U if f_preds[i][+1] > 0.5) / len(self.U)
            
            # Check convergence
            if loss.item() >= prev_loss:
                break
            prev_loss = loss.item()
        return self.theta, self.pi_p_est


# Add a GNN model for the classifier
