from sklearn.linear_model import LinearRegression
from benchmark.template_sklearn_cv import train_and_eval
if __name__ == '__main__':

    train_and_eval("../config/linear.yaml", LinearRegression(), "linear_regression")