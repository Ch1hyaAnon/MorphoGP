import subprocess

models = [
    "linear_regression",
    "random_forest",
    "polynomial_regression",
    "svr",
    "knn",
    "mlp",
    "transformer"
]

for m in models:
    print(f"\n=== Running {m} ===")
    subprocess.run(["python", f"models/{m}.py"])