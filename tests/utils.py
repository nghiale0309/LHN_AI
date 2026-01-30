import time
import numpy as np

def run_and_time(func):
    start = time.perf_counter()
    result = func()
    return result, time.perf_counter() - start

def repeat_median(func, n=5):
    times = []
    for _ in range(n):
        _, t = run_and_time(func)
        times.append(t)
    return float(np.median(times))

def speed_report(sk_time, lhn_time):
    eps = 1e-9
    ratio = sk_time / max(lhn_time, eps)
    if ratio > 1:
        return f"LHN_AI chạy nhanh hơn Scikit-learn gấp {ratio:.2f} lần"
    else:
        return f"LHN_AI chạy chậm hơn Scikit-learn gấp {1/ratio:.2f} lần"
