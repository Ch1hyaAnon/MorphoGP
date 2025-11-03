import yaml
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from benchmark.template_sklearn_cv import train_and_eval

with open("../config/polynomial.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = Pipeline([
    ("poly", PolynomialFeatures(degree=cfg["model"]["degree"])),
    ("lin", LinearRegression())
])

train_and_eval("../config/polynomial.yaml", model, "polynomial_regression")