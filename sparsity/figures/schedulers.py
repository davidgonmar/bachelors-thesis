import matplotlib.pyplot as plt
import numpy as np

from compress.sparsity.schedulers import (
    linear_sparsity,
    poly_sparsity,
    cos_sparsity,
    exp_sparsity,
)

max_iter = 10
target_sparsity = 0.8
iters = np.linspace(0, max_iter, 500)

linear_vals = linear_sparsity(iters, max_iter, target_sparsity)
poly_vals = poly_sparsity(iters, max_iter, target_sparsity)
cos_vals = cos_sparsity(iters, max_iter, target_sparsity)
exp_vals = exp_sparsity(iters, max_iter, target_sparsity)

plt.figure()
plt.plot(iters, linear_vals, label="Linear")
plt.plot(iters, poly_vals, label="Polynomial (β=0.5)")
plt.plot(iters, cos_vals, label="Cosine")
plt.plot(iters, exp_vals, label="Exponential (α=5)")
plt.xlabel("Iteration")
plt.ylabel("Sparsity (fraction pruned)")
plt.grid(True)
plt.legend()
plt.tight_layout()

output_path = "./sparsity_schedule_fns.pdf"
plt.savefig(output_path)

plt.show()
