from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --- regularization classes ---
class Normal:
    """No regularization - lets plain linear regression go through the
    same LinearRegression class as Lasso/Ridge."""
    def __init__(self, l=0):
        self.l = l
    def __call__(self, theta):
        return 0
    def derivation(self, theta):
        return 0


class Lasso:
    def __init__(self, l):
        self.l = l
    def __call__(self, theta):  # __call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
    def derivation(self, theta):
        return self.l * np.sign(theta)


class Ridge:
    def __init__(self, l):
        self.l = l
    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))
    def derivation(self, theta):
        return self.l * 2 * theta


class Elastic:
    def __init__(self, l, l_ratio=0.5):
        self.l = l
        self.l_ratio = l_ratio
    def __call__(self, theta):
        l1 = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2 = (1 - self.l_ratio) * self.l * np.sum(np.square(theta))
        return l1 + l2
    def derivation(self, theta):
        l1 = self.l_ratio * self.l * np.sign(theta)
        l2 = 2 * self.l * (1 - self.l_ratio) * theta
        return l1 + l2


# --- main class ---
class LinearRegression(object):

    kfold = KFold(n_splits=5)

    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=500,
                 bs=50, cv=kfold, init_method='zeros', use_momentum=False,
                 momentum=0.9):
        self.lr = lr
        self.num_epochs = num_epochs
        self.bs = bs
        self.method = method
        self.cv = cv
        self.regularization = regularization

        # ---- Task 1 additions ----
        self.init_method = init_method       # 'zeros' or 'xavier'
        self.use_momentum = use_momentum     # bool
        self.momentum = momentum             # in (0, 1)

    # mse
    def mse(self, ytrue, ypred):
        # ytrue shape: (m, ) ==> m = number of samples
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    # Task 1: r2 score
    def r2(self, ytrue, ypred):
        ss_res = np.sum((ytrue - ypred) ** 2)
        ss_tot = np.sum((ytrue - np.mean(ytrue)) ** 2)
        return 1 - (ss_res / ss_tot)

    # Task 1: weight initialization (zeros or xavier)
    def _initialize_theta(self, n):
        # n = number of inputs (features)
        if self.init_method == 'xavier':
            lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n))
            numbers = np.random.rand(n)
            return lower + numbers * (upper - lower)
        else:  # zeros
            return np.zeros(n)

    # fit
    def fit(self, X_train, y_train):

        # create a list for keeping kfold scores
        self.kfold_scores = list()
        self.kfold_r2_scores = list()

        # variable to know our loss is not improving anymore
        self.val_loss_old = np.inf

        # cross validation
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):

            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]

            self.theta = self._initialize_theta(X_cross_train.shape[1])
            self.prev_step = np.zeros_like(self.theta)  # for momentum

            for epoch in range(self.num_epochs):

                # Shuffle the data a little so that order does not impact our model
                perm = np.random.permutation(X_cross_train.shape[0])
                X_cross_train = X_cross_train[perm]
                y_cross_train = y_cross_train[perm]

                if self.method == 'mini':
                    for batch_idx in range(0, X_cross_train.shape[0], self.bs):
                        X_method_train = X_cross_train[batch_idx:batch_idx + self.bs]
                        y_method_train = y_cross_train[batch_idx:batch_idx + self.bs]
                        train_loss = self._train(X_method_train, y_method_train)
                elif self.method == 'sto':
                    for i in range(X_cross_train.shape[0]):
                        X_method_train = X_cross_train[i:i + 1]
                        y_method_train = y_cross_train[i:i + 1]
                        train_loss = self._train(X_method_train, y_method_train)
                else:  # batch
                    X_method_train = X_cross_train
                    y_method_train = y_cross_train
                    train_loss = self._train(X_method_train, y_method_train)

                yhat_val = self.predict(X_cross_val)
                val_loss_new = self.mse(y_cross_val, yhat_val)

                # early stopping
                if np.allclose(val_loss_new, self.val_loss_old):
                    break
                self.val_loss_old = val_loss_new

            val_r2_new = self.r2(y_cross_val, yhat_val)
            self.kfold_scores.append(val_loss_new)
            self.kfold_r2_scores.append(val_r2_new)
            print(f"Fold {fold}: mse={val_loss_new:.4f}, r2={val_r2_new:.4f}")

    # train
    def _train(self, X, y):
        # X shape: (m, n); y shape: (m, ); theta shape: (n, )

        # 1. predict
        yhat = self.predict(X)

        # 2. grad
        m = X.shape[0]
        grad = (1 / m) * X.T @ (yhat - y) + self.regularization.derivation(self.theta)

        # gradient clipping for numerical stability (large lr + momentum + sto
        # can otherwise overflow)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 5.0:
            grad = grad * (5.0 / grad_norm)

        # 3. update
        step = self.lr * grad
        if self.use_momentum:
            new_theta = self.theta - step + self.momentum * self.prev_step
        else:
            new_theta = self.theta - step

        if np.all(np.isfinite(new_theta)):
            self.theta = new_theta
        self.prev_step = step

        # return
        return self.mse(y, yhat)

    # predict
    def predict(self, X):
        return X @ self.theta  # (m, n) @ (n, ) = (m, )  <===== y

    # get theta
    def _coef(self):
        return self.theta[1:]

    # get bias
    def _bias(self):
        return self.theta[0]

    # Task 1: feature importance plot
    def plot_feature_importance(self, feature_names=None, top_n=15):
        """
        Plots feature importance based on |coefficient|. Assumes the
        intercept is the first entry in theta, and that features were
        scaled prior to fitting so magnitudes are comparable.
        """
        coefs = self._coef()

        if feature_names is None:
            feature_names = [f'x{i}' for i in range(len(coefs))]

        importance = np.abs(coefs)
        order = np.argsort(importance)[::-1][:top_n]

        plt.figure(figsize=(8, 6))
        plt.barh([feature_names[i] for i in order][::-1], importance[order][::-1])
        plt.xlabel('|coefficient|')
        plt.title('Feature importance (coefficient magnitude)')
        plt.tight_layout()
        plt.show()

        return dict(zip([feature_names[i] for i in order], importance[order]))


# --- convenience subclasses (inherit LinearRegression) ---
class NormalRegression(LinearRegression):
    def __init__(self, method, lr, **kwargs):
        self.regularization = Normal()
        super().__init__(self.regularization, lr, method, **kwargs)


class LassoRegression(LinearRegression):
    def __init__(self, method, lr, l, **kwargs):
        self.regularization = Lasso(l)
        super().__init__(self.regularization, lr, method, **kwargs)


class RidgeRegression(LinearRegression):
    def __init__(self, method, lr, l, **kwargs):
        self.regularization = Ridge(l)
        super().__init__(self.regularization, lr, method, **kwargs)


class ElasticRegression(LinearRegression):
    def __init__(self, method, lr, l, l_ratio=0.5, **kwargs):
        self.regularization = Elastic(l, l_ratio)
        super().__init__(self.regularization, lr, method, **kwargs)