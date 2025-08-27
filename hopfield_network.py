# -*- coding: utf-8 -*-
"""
Hopfield Network implementation for associative memory.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

class HopfieldNetwork(object):
   
    def train_weights(self, train_data):
        
        print("Start to train weights...")
        num_patterns = len(train_data)
        num_neurons = train_data[0].shape[0]
        self.num_neuron = num_neurons

        W = np.zeros((num_neurons, num_neurons))
        rho = np.sum([np.sum(pattern) for pattern in train_data]) / (num_patterns * num_neurons)

        for pattern in tqdm(train_data):
            centered = pattern - rho
            W += np.outer(centered, centered)

        np.fill_diagonal(W, 0)
        W /= num_patterns
        self.W = W

    def predict(self, data, num_iter=20, threshold=0, asyn=False):

        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn

        copied_data = np.copy(data)
        return [self._run(pattern) for pattern in tqdm(copied_data)]

    def _run(self, state):
        
        s = state
        e = self.energy(s)

        if not self.asyn:
            # Synchronous update
            for _ in range(self.num_iter):
                s = np.sign(self.W @ s - self.threshold)
                e_new = self.energy(s)
                if e == e_new:
                    break
                e = e_new
            return s
        else:
            # Asynchronous update
            for _ in range(self.num_iter):
                for _ in range(100):
                    idx = np.random.randint(0, self.num_neuron)
                    s[idx] = np.sign(self.W[idx].T @ s - self.threshold)
                e_new = self.energy(s)
                if e == e_new:
                    break
                e = e_new
            return s

    def energy(self, s):
       
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()