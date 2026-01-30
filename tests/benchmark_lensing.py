import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc
import lhn_AI.core_backend as lhn


np.random.seed(42)

N_TRAIN = 1000
N_TEST = 1000
EPOCHS = 2000
HIDDEN_DIM = 64
LR_LHN = 1e-4
SIGMA = 15.0

def phi_true(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def source_f(x, y):
    return -2.0 * (np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)

def compute_r2(pred, target):
    diff = pred - target
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((target - np.mean(target))**2)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot

X_train = np.random.uniform(-1.0, 1.0, size=(N_TRAIN, 2)).astype(np.float64)
X_test = np.random.uniform(-1.0, 1.0, size=(N_TEST, 2)).astype(np.float64)

f_train = source_f(X_train[:, 0], X_train[:, 1]).astype(np.float64)
f_test = source_f(X_test[:, 0], X_test[:, 1]).astype(np.float64)

phi_train = phi_true(X_train[:, 0], X_train[:, 1]).astype(np.float64)
phi_test = phi_true(X_test[:, 0], X_test[:, 1]).astype(np.float64)

gc.collect()

t0 = time.perf_counter()

net_lhn = lhn.SirenPhysicsNet([2, HIDDEN_DIM, HIDDEN_DIM, 1], SIGMA)
trainer_lhn = lhn.PhysicsTrainer(
    net_lhn,
    0.0,
    1.0,
    LR_LHN
)

lhn.train_batch_poisson(
    trainer_lhn,
    X_train,
    f_train,
    EPOCHS
)

t_train_lhn = time.perf_counter() - t0

t0 = time.perf_counter()
phi_preds_test = np.array([net_lhn.forward(float(x[0]), float(x[1]))[0] for x in X_test])
t_infer_lhn = time.perf_counter() - t0

r2_phi = compute_r2(phi_preds_test, phi_test)

print(f"Train Time: {t_train_lhn:.2f}s")
print(f"Potential R2 (test): {r2_phi:.4f}")

plt.figure(figsize=(12, 5))
idx = np.argsort(X_test[:, 0])
mask = np.abs(X_test[idx, 1]) < 0.05

X_plot = X_test[idx][mask, 0]
Phi_true_plot = phi_test[idx][mask]
Phi_pred_plot = phi_preds_test[idx][mask]

plt.plot(X_plot, Phi_true_plot, 'k-', lw=2, alpha=0.7)
plt.plot(X_plot, Phi_pred_plot, 'r--', lw=1.5)

plt.title("Poisson PINN Solution φ(x, y≈0)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
