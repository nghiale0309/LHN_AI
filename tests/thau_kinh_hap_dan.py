import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from lhn_AI.core_backend import SirenPhysicsNet, PhysicsTrainer, train_batch_poisson

np.random.seed(1500)

EPOCHS_PER_CHUNK = 100
TOTAL_CHUNKS = 5
BATCH_SIZE = 1000
LIMIT = 10.0
LR = 5e-3
SIGMA = 0.5
AMP = 20.0

def gaussian_kappa(x, y):
    r2 = x**2 + y**2
    return AMP * np.exp(-r2 / (2 * SIGMA**2))

def theoretical_einstein_radius():
    r = np.linspace(1e-3, LIMIT, 50000)
    k = AMP * np.exp(-r**2 / (2 * SIGMA**2))
    m = np.cumsum(k * 2 * np.pi * r * (r[1] - r[0]))
    a = m / (np.pi * r)
    return r[np.argmin(np.abs(a - r))]

def sample_points(n):
    n1 = int(n * 0.5) 
    n2 = n - n1
    u = np.random.uniform(-LIMIT, LIMIT, (n1, 2))
    g = np.random.randn(n2, 2) * (2.0 * SIGMA)
    x = np.vstack([u, g])
    return np.clip(x, -LIMIT, LIMIT)

def radial_integrated_deflection(R, lap, bins):
    r_mid = 0.5 * (bins[1:] + bins[:-1])
    alpha = np.zeros_like(r_mid)
    for i, r in enumerate(r_mid):
        m = (R <= r)
        if np.sum(m) > 0:
            mass = (np.mean(lap[m])) * (np.pi * r**2)
            alpha[i] = mass / (np.pi * r)
    return r_mid, alpha

def main():
    net = SirenPhysicsNet([2, 128, 128, 128, 1], 30.0)
    trainer = PhysicsTrainer(net, 0.0, 1.0, LR)

    print(f"--- TRAINING ({TOTAL_CHUNKS * EPOCHS_PER_CHUNK} Epochs) ---")
    for i in range(TOTAL_CHUNKS):
        X = sample_points(BATCH_SIZE)
        k = gaussian_kappa(X[:, 0], X[:, 1])
        train_batch_poisson(trainer, X, k, EPOCHS_PER_CHUNK)
        
        if (i+1) % 10 == 0: 
            print(f"Progress: {(i+1)/TOTAL_CHUNKS*100:.0f}%")

    r_th = theoretical_einstein_radius()

    N = 400
    x = np.linspace(-LIMIT, LIMIT, N)
    y = np.linspace(-LIMIT, LIMIT, N)
    XX, YY = np.meshgrid(x, y)
    lap = np.zeros_like(XX)
    
    for i in range(N):
        for j in range(N):
            lap[i, j] = net.forward(XX[i, j], YY[i, j])[3]

    R = np.sqrt(XX**2 + YY**2)
    bins = np.linspace(0.1, LIMIT, 500)
    r_prof, alpha_prof = radial_integrated_deflection(R, lap, bins)
    
    idx_min = np.argmin(np.abs(alpha_prof - r_prof))
    r_nn = r_prof[idx_min]
    
    acc = 100 * (1 - abs(r_nn - r_th) / r_th)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor("black")

    im = ax.imshow(
        lap, extent=[-LIMIT, LIMIT, -LIMIT, LIMIT], origin="lower",
        cmap="inferno", interpolation="bicubic",
        vmin=0, vmax=np.percentile(lap, 99.5), 
        alpha=1.0
    )

    bh_radius = r_th * 0.3
    bh_circle = Circle((0, 0), bh_radius, color='black', zorder=20)
    ax.add_patch(bh_circle)
    bh_edge = Circle((0, 0), bh_radius, color='white', fill=False, linewidth=1.5, alpha=0.8, zorder=21)
    ax.add_patch(bh_edge)
    ax.text(0, 0, "Black Hole", color="white", ha="center", va="center", fontsize=9, fontweight='bold', zorder=22)

    theta = np.linspace(0, 2*np.pi, 2000)
    colors = cm.cool(theta / (2*np.pi))
    for width, alpha_val in [(12, 0.05), (8, 0.1), (4, 0.3)]:
        r_vary = r_th + np.random.uniform(-0.02, 0.02, len(theta)) * r_th
        x_th = r_vary * np.cos(theta)
        y_th = r_vary * np.sin(theta)
        ax.scatter(x_th, y_th, s=width**2, c=colors, alpha=alpha_val, linewidth=0, zorder=15)
    
    ax.plot(r_th * np.cos(theta), r_th * np.sin(theta), color='cyan', linewidth=1, alpha=0.5, zorder=16)

    N_stars = 250
    theta_s = np.random.uniform(0, 2*np.pi, N_stars)
    r_s = r_nn + np.random.uniform(-0.4, 0.4, N_stars)
    x_s = r_s * np.cos(theta_s)
    y_s = r_s * np.sin(theta_s)
    sizes_s = np.random.uniform(20, 180, N_stars)
    alphas_s = np.random.uniform(0.5, 1.0, N_stars)
    
    ax.scatter(
        x_s, y_s, color='gold', marker='*', s=sizes_s, alpha=alphas_s, 
        edgecolors='orange', linewidth=0.5, zorder=18
    )

    legend_elements = [
        Line2D([0], [0], color='cyan', linewidth=3, alpha=0.8, label='Theory (Aurora Ring)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Neural Net (Stars)')
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor="black", labelcolor="white", fontsize=11, framealpha=0.6)

    stats_text = (
        f"Theory Radius: {r_th:.4f}\n"
        f"NN Radius:     {r_nn:.4f}\n"
        f"Accuracy:      {acc:.2f}%"
    )
    text_color = "lime" if acc > 0 else "red"
    ax.text(-LIMIT*0.95, LIMIT*0.85, stats_text, color=text_color, fontsize=12, family="monospace",
            bbox=dict(facecolor="black", edgecolor=text_color, alpha=0.8, boxstyle='round,pad=0.8'))
    
    ax.set_title(f"GRAVITATIONAL LENSING SIMULATION (SIREN)", color="white", fontsize=16, pad=20, fontweight='bold')
    ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()