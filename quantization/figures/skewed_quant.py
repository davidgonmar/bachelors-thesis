import math
import torch
from compress.quantization.calibrate import calibrate
from compress.quantization.common import (
    quantize,
    dequantize,
    IntAffineQuantizationMode,
    IntAffineQuantizationSpec,
)
import matplotlib.pyplot as plt


def skew_normal(size, loc=0.0, scale=1.0, skew=0.3):
    # ensure `size` is a tuple
    size = (size,) if isinstance(size, int) else tuple(size)

    # compute δ = α / sqrt(1 + α²) using math for the constants
    delta = skew / math.sqrt(1.0 + skew**2)

    # draw two independent standard normals
    u0 = torch.randn(size)
    u1 = torch.randn(size)

    # construct skew‐normal variate
    z = delta * u0.abs() + math.sqrt(1.0 - delta**2) * u1

    return loc + scale * z


# sample 10k skewed normals
x_skewed = skew_normal(10000, loc=0.0, scale=1.0, skew=0.3)

# calibrate & quantize
spec = IntAffineQuantizationSpec(
    nbits=2, signed=True, quant_mode=IntAffineQuantizationMode.SYMMETRIC
)
qparams = calibrate(x_skewed, spec)
quantized = dequantize(quantize(x_skewed, qparams), qparams)

# plot
plt.hist(x_skewed.numpy(), bins=100, alpha=0.5, label="Skewed Normal")
plt.hist(quantized.numpy(), bins=100, alpha=0.5, label="Quantized")
plt.legend()
plt.show()
