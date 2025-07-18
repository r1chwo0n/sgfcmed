"""
SGFCMedParallel: String Grammar Fuzzy C-Medians with multiprocessing
Author: Atcharin Klomsae
Date: 23/4/2020

Description:
This implementation performs fuzzy clustering on strings using Levenshtein distance
and modified median string updates. It uses multiprocessing to improve speed on large datasets.
"""

import random
import multiprocessing
from functools import lru_cache, partial

@lru_cache(maxsize=None) 
def lev(s1, s2):
    # Initialize distance matrix
    len_s1, len_s2 = len(s1), len(s2)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

    # Base cases: distance from empty string
    for i in range(len_s1 + 1): dp[i][0] = i
    for j in range(len_s2 + 1): dp[0][j] = j

    # Fill in the matrix
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost   # substitution
            )
           
    return dp[len_s1][len_s2]

def parallel_total_distance(args):
    """
    Wrapper for multiprocessing: returns (candidate, total weighted distance).
    args = (candidate, S, U, m)
    """
    candidate, S, U, m = args
    total = sum((U[k] ** m) * lev(candidate, S[k]) for k in range(len(S)))
    return candidate, total

class SGFCMedParallel:
    def __init__(self, C=2, m=2.0, max_iter=100, tol=1e-4):
        """
        Initialize clustering model.

        Parameters:
            C (int): Number of clusters
            m (float): Fuzzifier (greater than 1)
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for convergence
        """
        self.C = C  # number of clusters
        self.m = m  # fuzzifier for controlling fuzziness
        self.max_iter = max_iter  # maximum number of training iterations
        self.tol = tol  # convergence threshold
        self.prototypes_ = []  # list of prototype strings (cluster centers)
        self.U_ = []  # fuzzy membership matrix

        random.seed(42)  # ensure reproducible random initialization

        # Prepare Levenshtein distance function with memoization
        self._levenshtein = self._memoized_levenshtein 


    @lru_cache(maxsize=None) # helps in reducing the execution time of the function by using memoization technique.
    def _memoized_levenshtein(self,s1,s2):
        """
        Create a memoized version of Levenshtein distance to avoid redundant computation.
        """
        return lev(s1, s2)

    def _total_distance(self, candidate, S):
        """
        Compute the total distance from a candidate string to all strings in S.
        Used for finding the best prototype.
        """
        return sum(self._levenshtein(candidate, s) for s in S)

    def _update_prototype(self, s, S, m, U, alphabet):
        for i in range(len(s)):
            candidates = [s] # include current string as candidate

            # Generate substitution candidates
            for a in alphabet:
                if i < len(s) and a != s[i]:
                    candidates.append(s[:i] + a + s[i+1:]) 
            
            # Generate deletion candidate
            if len(s) > 1:
                candidates.append(s[:i] + s[i+1:]) 
            
            # Generate insertion candidates
            for a in alphabet:
                candidates.append(s[:i] + a + s[i:]) 

            args = [(cand, S, U, m) for cand in candidates]

            # Use multiprocessing to compute total distances for all candidates
            with multiprocessing.Pool() as pool:
                results = pool.map(parallel_total_distance, args)

                # Select the candidate with minimum total distance as new prototype
                s = min(results, key=lambda x: x[1])[0] 

        return s

    def fit(self, S):
        """
        Train the model on list of strings S using fuzzy c-means clustering.
        """
        N = len(S)  # number of strings
        alphabet = set(''.join(S))  # all unique characters in the dataset

        # Step 1: Initialize fuzzy membership matrix randomly
        self.U_ = [[random.random() for _ in range(self.C)] for _ in range(N)] 
        for i in range(N):
            total = sum(self.U_[i]) 
            self.U_[i] = [u / total for u in self.U_[i]] 

        # Step 2: Initialize cluster prototypes randomly from input strings
        self.prototypes_ = random.sample(S, self.C)

        for iteration in range(self.max_iter):
            old_prototypes = self.prototypes_[:]

            # Step 3: Update cluster prototypes using modified median
            for i in range(self.C):        
                U = [self.U_[k][i] for k in range(N)]      
                self.prototypes_[i] = self._update_prototype(self.prototypes_[i], S, self.m, U, alphabet)

            # Step 4: Update fuzzy membership values based on distance to prototypes
            for k in range(N):  # for each string
                for i in range(self.C):  # for each cluster
                    d_i = self._levenshtein(S[k], self.prototypes_[i]) + 1e-6  # avoid division by 0
                    denom = sum(
                        (d_i / (self._levenshtein(S[k], self.prototypes_[j]) + 1e-6)) ** (2 / (self.m - 1))
                        for j in range(self.C)
                    )
                    self.U_[k][i] = 1 / (denom + 1e-6)

            # Step 5: Check for convergence (change in prototypes)
            change = sum(self._levenshtein(p1, p2) for p1, p2 in zip(old_prototypes, self.prototypes_))
            if change < self.tol:
                break

    def membership(self):
        """
        Return the current fuzzy membership matrix.
        """
        return self.U_

    def prototypes(self):
        """
        Return the current list of cluster prototype strings.
        """
        return self.prototypes_

    def predict(self, S):
        """
        Assign each string in S to the nearest prototype (crisp clustering).
        """
        preds = []
        for s in S:
            distances = [self._levenshtein(s, proto) for proto in self.prototypes_]
            preds.append(distances.index(min(distances)))  # cluster with min distance
        return preds