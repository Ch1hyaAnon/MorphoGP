import yaml
from sklearn.neighbors import KNeighborsRegressor
from benchmark.template_sklearn_cv import train_and_eval

with open("../config/knn.yaml", "r") as f:
    cfg = yaml.safe_load(f)
model = KNeighborsRegressor(**cfg["model"])
train_and_eval("../config/knn.yaml", model, "knn")