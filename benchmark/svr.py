import yaml
from sklearn.svm import SVR
from benchmark.template_sklearn_cv import train_and_eval

with open("../config/svr.yaml", "r") as f:
    cfg = yaml.safe_load(f)
model = SVR(**cfg["model"])
train_and_eval("../config/svr.yaml", model, "svr")