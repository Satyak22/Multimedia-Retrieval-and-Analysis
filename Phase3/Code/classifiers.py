#! /usr/bin/env python3
import cvxopt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy as np

def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Leaf_Node:

    def __init__(self, rows):
        self.future_pred = list(class_counts(rows).keys())[0]

class D_Node:
    def __init__(self,
                 rule,
                 left_br,
                 right_br):
        self.rule = rule
        self.left_br = left_br
        self.right_br = right_br


class rule:
    def __init__(self, attr, value):
        self.attr = attr
        self.value = value

    def match(self, row):
        val = row[self.attr]
        if isinstance(self.value, int) or isinstance(self.value, float):
            return val >= self.value
        else:
            return val == self.value


class DecisionTreeClassifier():

    def __init__(self):
        self.root = None


    def split(self, rows):
        current_gain, uncertainty, counts, good_ques, best_gain = 0, 1, class_counts(rows), None, 0
        for count in counts:
            prob = counts[count] / float(len(rows))
            uncertainty -= prob ** 2

        for col in range(len(rows[0]) - 1):
            values = set([row[col] for row in rows])
            for val in values:
                ques = rule(col, val)

                left_rows, right_rows = [], []
                for row in rows:
                    if ques.match(row):
                        left_rows.append(row)
                    else:
                        right_rows.append(row)

                if len(left_rows) == 0 or len(right_rows) == 0:
                    continue

                gl_p = float(len(left_rows)) / (len(left_rows) + len(right_rows))
                ginileft, giniright = 1, 1
                countsleft, countsright = class_counts(left_rows), class_counts(right_rows)
                for count in countsleft:
                    prob = countsleft[count] / float(len(left_rows))
                    ginileft -= prob ** 2
                for count in countsright:
                    prob1 = countsright[count] / float(len(right_rows))
                    giniright -= prob1 ** 2
                gain = uncertainty - gl_p * ginileft - (1 - gl_p) * giniright

                if gain >= best_gain:
                    best_gain, good_ques = gain, ques

        return best_gain, good_ques


    def make_tree(self, rows):
        gain, rule = self.split(rows)
        if gain == 0:
            return Leaf_Node(rows)


        left_rows, right_rows = [], []


        for row in rows:
            if rule.match(row):
                left_rows.append(row)
            else:
                right_rows.append(row)

        left_br, right_br = self.make_tree(left_rows), self.make_tree(right_rows)
        return D_Node(rule, left_br, right_br)


    def predict(self, row, node):
        if isinstance(node, Leaf_Node):
            return node.future_pred

        if node.rule.match(row):
            return self.predict(row, node.left_br)
        else:
            return self.predict(row, node.right_br)

    def fit(self, data):
        self.root = self.make_tree(data)

    def transform(self, data):
        results = []
        for row in data:
            results.append(self.predict(row, self.root))
        return results


def rbf_kernel(gamma):
    return lambda x, y: np.exp(-gamma*np.linalg.norm(np.subtract(x, y)))

class SupportVectorMachine(object):
    """The Support Vector Machine classifier.
    """
    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None
        self.bias = 0.0
        self.w = None


    def fit(self, X, y):
        n_samples, n_features = np.shape(X) # n_samples - 198 n_features - 20

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(self.gamma)
        # Calculate kernel matrix
        K = np.zeros((n_samples, n_samples))            # K
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])


        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.C)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)            # solution

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])                        #a
        print(lagr_mult)

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        #sv = (lagr_mult > 1e-5).astype('int')
        sv = (lagr_mult > 1e-1)

        self.lagr_multipliers = lagr_mult[sv]
        # Get the samples that will act as support vectors
        self.support_vectors = X[sv]

        self.support_vector_labels = y[sv]

        print("%d support vectors out of %d points" % (len(lagr_mult), n_samples))
        self.bias = np.mean([y_k - self.calculate_bias(x_k) for (y_k, x_k) in zip(self.support_vector_labels, self.support_vectors)])
        print("Bias is", self.bias)

    def calculate_bias(self, x):
        result = self.bias
        for z_i, x_i, y_i in zip(self.lagr_multipliers,
                self.support_vectors,
                self.support_vector_labels):
            result += z_i * y_i * self.kernel(x_i, x)
        return np.sign(result).item()

    def predict(self, x):
        results = []
        for val in x:
            result = self.bias
            for z_i, x_i, y_i in zip(self.lagr_multipliers,
                 self.support_vectors,
                 self.support_vector_labels):
                result += z_i * y_i * self.kernel(x_i, val)
            results.append(np.sign(result).item())
        return results


