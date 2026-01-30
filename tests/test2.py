import numpy as np
import time
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.metrics import accuracy_score, log_loss
from lhn_AI import LogisticRegression

np.random.seed(36)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def measure_time(fn, repeat=3):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return np.median(times)


sizes = [10_000, 100_000, 500_000]

for n in sizes:
    X = np.random.randn(n, 1)
    logits = 3.0 * X.flatten() - 1.5
    probs = sigmoid(logits)
    y = (probs > 0.5).astype(int)

    sk_model = SKLogistic(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=1
    )

    lhn_model = LogisticRegression(
        lr=0.1,
        max_iter=1000
    )

    sk_time = measure_time(lambda: sk_model.fit(X, y))
    lhn_time = measure_time(lambda: lhn_model.fit(X, y))

    sk_prob = sk_model.predict_proba(X)[:, 1]
    lhn_prob = lhn_model.predict_proba(X)

    sk_acc = accuracy_score(y, sk_prob > 0.5)
    lhn_acc = accuracy_score(y, lhn_prob > 0.5)

    sk_loss = log_loss(y, sk_prob)
    lhn_loss = log_loss(y, lhn_prob)

    speed = sk_time / lhn_time if lhn_time > 0 else float("inf")

    print(f"\nDataset size: {n}")
    print(f"Scikit-learn  time={sk_time:.6f}s  acc={sk_acc:.5f}  logloss={sk_loss:.6f}")
    print(f"LHN_AI        time={lhn_time:.6f}s  acc={lhn_acc:.5f}  logloss={lhn_loss:.6f}")

    if speed >= 1:
        print(f"LHN_AI nhanh hơn Scikit-learn gấp {speed:.2f} lần")
    else:
        print(f"LHN_AI chậm hơn Scikit-learn gấp {1/speed:.2f} lần")
