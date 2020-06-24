from project_utils import *
import argparse
import cvxopt
from task1 import populate_database


class SupportVectorMachine(object):
    """The Support Vector Machine classifier.
    """
    def __init__(self, C=1, kernel=rbf_kernel, power=3, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None


    def fit(self, X, y):

        n_samples, n_features = np.shape(X) # n_samples - 198 n_features - 20

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))            # K
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if self.C is None:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)            # solution

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])                        #a

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        sv = lagr_mult > 1e-5
        #sv = (lagr_mult > 1e-10).astype('int')
        ind = np.arange(len(lagr_mult))[sv]
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[sv]
        # Get the samples that will act as support vectors
        self.support_vectors = X[sv]
        # Get the corresponding labels
        # print(y)
        # print(sv)
        self.support_vector_labels = y[sv]
        print(self.support_vector_labels.shape)                      # sv_y
        print("%d support vectors out of %d points" % (len(lagr_mult), n_samples))

        #Calculate intercept with first support vector
        self.intercept = 0;
        for n in range (len(self.lagr_multipliers)):
            self.intercept += self.support_vector_labels[n]
            self.intercept -= np.sum(self.lagr_multipliers * self.support_vector_labels * kernel_matrix[ind[n], sv])
        self.intercept /= len(self.lagr_multipliers)


        # self.intercept = self.support_vector_labels[0]
        # for i in range(len(self.lagr_multipliers)):
        #     self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
        #         i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    # def predict(self, X):
    #     y_pred = []
    #     # Iterate through list of samples and make predictions
    #     for sample in X:
    #         prediction = 0
    #         # Determine the label of the sample by the support vectors
    #         for i in range(len(self.lagr_multipliers)):
    #             prediction += self.lagr_multipliers[i] * self.support_vector_labels[
    #                 i] * self.kernel(self.support_vectors[i], sample)
    #         prediction += self.intercept
    #         y_pred.append(prediction)
    #     return np.array(y_pred)

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.lagr_multipliers, self.support_vector_labels, self.support_vectors):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict + self.intercept

    def predict(self, X):
        return np.sign(self.project(X))
    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(prediction)
        return np.array(y_pred)

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", help="Path to the folder containing training images", type=str,  required=True)
    parser.add_argument("--train_metadata", help="Path to the training metadata", type=str, required=True)
    parser.add_argument("--test_folder", help="Path to the folder containing the test images", type=str, required=True)
    parser.add_argument("--classifier", help="Select classifier - SVM, DT or PPR", type=str, choices=["SVM", "DT", "PPR"], required=True)
    return parser

if __name__ == "__main__":
    parser = setup_arg_parse()
    args = parser.parse_args()
    populate_database(args)
    mongo_client = connect_to_db()
    #filterstring = convert_label_to_filterstring(args.label)

    dorsal_data_matrix, _ = get_data_matrix("HOG", convert_label_to_filterstring("dorsal"))

    palmar_data_matrix, _ = get_data_matrix("HOG", convert_label_to_filterstring("palmar"))

    dorsal_labels = np.array([1]*dorsal_data_matrix.shape[0])
    palmar_labels = np.array([-1] * palmar_data_matrix.shape[0])

    labels = np.append(dorsal_labels, palmar_labels, axis = 0)

    concatenated_data = np.append(dorsal_data_matrix, palmar_data_matrix, axis = 0)
    concatenated_labels = np.concatenate((dorsal_labels.flatten(), palmar_labels.flatten())).flatten()

    reduced_data = reduce_dimensions_lda(concatenated_data, 500)

    print(dorsal_data_matrix.shape)
    print(palmar_data_matrix.shape)
    print(dorsal_labels.shape)
    print(palmar_labels.shape)
    print(reduced_data.shape)
    print(concatenated_labels.shape)
    print(concatenated_labels)

    train_concatenated_data = reduced_data[-170:-1]
    train_concatenated_labels = concatenated_labels[-170:-1]

    # print(train_concatenated_labels)

    clf = SupportVectorMachine(kernel=rbf_kernel, power=4, coef=1)
    clf.fit(train_concatenated_data, train_concatenated_labels)



    values = clf.predict(reduced_data[0:15])
    print(values)

