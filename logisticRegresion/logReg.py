__author__ = 'Luis Fernando'

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator


class LogReg(BaseEstimator):

    def __init__(self, reg=0):
        self.reg = reg
        self.theta = []

    def fit(self, x, y):

        theta = np.zeros(x.shape[1] + 1, dtype=np.float64)
        x_1 = np.append( np.ones((x.shape[0], 1)), x, axis=1)
        self.theta = minimize(fun=self._cost, x0=theta, jac=self._grad, args=(x_1, y)).x

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-1.0 * x))

    def _cost(self, theta, x, y):
    # computes cost given predicted and actual values
        p = self._sigmoid(np.dot(x, theta))  # predicted probability of label 1
        p[np.where(p == 1)] = 0.99
        p[np.where(p == 0)] = 0.01
        m = y.size + 0.0
        log_l = (-y) * np.log(p) - (1.0 - y) * np.log(1.0 - p)  # log-likelihood vector
        reg_cost = self.reg/(2.0 * m) * np.sum(theta[1:] ** 2)
        return log_l.mean() + reg_cost

    def _grad(self, theta, x, y):
        p = self._sigmoid(np.dot(x, theta))
        error = p - y  # difference between label and prediction
        m = y.size + 0.0
        grad = np.dot(error, x)/m + np.dot(self.reg/m, theta.T)  # gradient vector
        return grad

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def predict_proba(self, x):
        res = np.zeros((x.shape[0], 2))
        x_1 = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        p_1 = self._sigmoid(np.dot(x_1, self.theta))
        res[:, 1] = p_1
        res[:, 0] = 1 - res[:, 1]
        return res

    def score(self, x, y):
         from sklearn.metrics import accuracy_score
         return accuracy_score(self.predict(x), y)

# Code to test
if __name__ == '__main__':
    from matplotlib  import pyplot
    from sklearn.metrics import accuracy_score
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.cross_validation import train_test_split

    data = np.loadtxt('ex2data1.txt', delimiter=',')

    # Plot
    X = data[:, 0:2]
    y = data[:, 2]

    pos = np.where(y == 1)
    neg = np.where(y == 0)
    pyplot.figure(1)
    pyplot.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    pyplot.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    pyplot.xlabel('Exam 1 score')
    pyplot.ylabel('Exam 2 score')
    pyplot.legend(['Not Admitted', 'Admitted'])
    pyplot.show()

    # Train - test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Cross validation
    clf = LogReg()
    tuned_parameters = {'reg': np.arange(0,2,0.1)}
    scores = ['accuracy', 'f1']

    for s in scores:
        print("# Tuning hyper-parameters for %s" % s)
        print("")

        clf = GridSearchCV(LogReg(reg=0), tuned_parameters, cv=5, scoring=s)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print("")
        print(clf.best_estimator_)
        print("")
        print("Grid scores on development set:")
        print("")
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print("")

        mean_score = [mean_score.mean_validation_score for mean_score in clf.grid_scores_]
        pyplot.figure()
        pyplot.plot(tuned_parameters['reg'], mean_score)
        pyplot.title('Cross validation: ' + s)
        pyplot.xlabel('Regularization')
        pyplot.ylabel(s)
        pyplot.show()

        print("Detailed classification report:")
        print("")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print("")
        y_pred = clf.predict(x_test)
        print(classification_report(y_test, y_pred))
        print("")

    # Wait and end
    raw_input("Press enter to continue")