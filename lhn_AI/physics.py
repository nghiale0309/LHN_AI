import math
import numpy as np

from . import (
    SirenPhysicsNet,
    PhysicsTrainer,
    LensingSampler
)


class LensingExperiment:
    def __init__(
        self,
        layers,
        w0=30.0,
        lr=1e-4,
        lambda_lens=1.0,
        lambda_poisson=1.0,
        total_steps=20000,
        batch_size=256
    ):
        self.net = SirenPhysicsNet(layers, w0)
        self.trainer = PhysicsTrainer(
            self.net,
            lambda_lens,
            lambda_poisson,
            lr
        )
        self.sampler = LensingSampler()
        self.total_steps = total_steps
        self.batch_size = batch_size

    def kappa_model(self, x, y):
        r = math.sqrt(x * x + y * y) + 1e-6
        return 0.5 / r

    def train_step(self, step):
        n_core = int(self.batch_size * 0.4)
        n_ring = int(self.batch_size * 0.4)
        n_far = self.batch_size - n_core - n_ring

        for _ in range(n_core):
            x, y = self.sampler.sample(0)
            kappa = self.kappa_model(x, y)
            self.trainer.step(x, y, kappa, step)

        if step > 0.2 * self.total_steps:
            for _ in range(n_ring):
                x, y = self.sampler.sample(1)
                kappa = self.kappa_model(x, y)
                self.trainer.step(x, y, kappa, step)

        for _ in range(n_far):
            x, y = self.sampler.sample(2)
            self.trainer.step(x, y, 0.0, step)

    def train(self, log_interval=500):
        for step in range(self.total_steps):
            self.train_step(step)

            if step % log_interval == 0:
                psi, dx, dy, lap = self.net.forward(0.1, 0.1)
                print(
                    f"[{step:06d}] "
                    f"psi={psi:.4e} "
                    f"grad=({dx:.3e},{dy:.3e}) "
                    f"lap={lap:.3e}"
                )

    def evaluate_grid(self, xmin=-2, xmax=2, n=256):
        xs = np.linspace(xmin, xmax, n)
        ys = np.linspace(xmin, xmax, n)

        psi = np.zeros((n, n))
        gx = np.zeros((n, n))
        gy = np.zeros((n, n))
        lap = np.zeros((n, n))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                v, dx, dy, l = self.net.forward(x, y)
                psi[j, i] = v
                gx[j, i] = dx
                gy[j, i] = dy
                lap[j, i] = l

        return {
            "psi": psi,
            "grad_x": gx,
            "grad_y": gy,
            "laplacian": lap,
            "x": xs,
            "y": ys,
        }
