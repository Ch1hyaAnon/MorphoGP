import yaml
from sklearn.ensemble import RandomForestRegressor
from benchmark.template_sklearn_cv import train_and_eval

with open("../config/random_forest.yaml", "r") as f:
    cfg = yaml.safe_load(f)
model = RandomForestRegressor(**cfg["model"])
train_and_eval("../config/random_forest.yaml", model, "random_forest")