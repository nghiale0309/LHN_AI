import numpy as np
from lhn_AI import LinearRegression
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.metrics import r2_score
from utils import repeat_median, speed_report

np.random.seed(42)

sizes = [10_000, 100_000, 1_000_000]

for n in sizes:
    X = np.random.rand(n, 1)
    noise = np.random.randn(n) * 0.5  
    y = 4.0 * X.flatten() + 10.0 + noise

    sk_model = SkLinearRegression()
    lhn_model = LinearRegression()

    sk_time = repeat_median(lambda: sk_model.fit(X, y))
    lhn_time = repeat_median(lambda: lhn_model.fit(X, y.reshape(-1, 1)))

    sk_pred = sk_model.predict(X)
    lhn_pred = X.flatten() * lhn_model.get_weight() + lhn_model.get_bias()

    sk_mse = np.mean((sk_pred - y) ** 2)
    lhn_mse = np.mean((lhn_pred - y) ** 2)

    sk_r2 = r2_score(y, sk_pred)
    lhn_r2 = r2_score(y, lhn_pred)

    print(f"\nDataset size: {n}")
    print(f"Scikit-learn  time={sk_time:.6f}s  MSE={sk_mse:.6f}  R2={sk_r2:.6f}")
    print(f"LHN_AI        time={lhn_time:.6f}s  MSE={lhn_mse:.6f}  R2={lhn_r2:.6f}")
    print(speed_report(sk_time, lhn_time))
