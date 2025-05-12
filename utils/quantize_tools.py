import torch

def quantize_fp32_to_fp16_fp32(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()  # float32 -> float16 -> float32

def quantize_fp32_to_fp8_fp32(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x_abs = x.abs()

    mantissa, exponent = torch.frexp(x_abs)  # x = mantissa * 2**exponent
    exponent = exponent + 14  # 15 bias - 1 (from frexp gives mantissa in [0.5, 1))

    # Too small or too big
    underflow = exponent <= 0
    overflow = exponent >= 31

    # 2-bit mantissa
    mantissa_scaled = (mantissa - 0.5) * 4
    mantissa_int = torch.clamp((mantissa_scaled + 0.5).int(), 0, 3)

    quantized = torch.pow(2.0, exponent - 15) * (1.0 + mantissa_int.float() / 4.0)
    quantized[underflow] = 0.0
    quantized[overflow] = 57344.0
    return quantized * sign

def quantize_fp32_to_ufp6_fp32(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    mantissa, exponent = torch.frexp(x)
    exponent = exponent + 6  # Bias 7 - 1

    underflow = exponent <= 0
    overflow = exponent >= 15

    mantissa_scaled = (mantissa - 0.5) * 4
    mantissa_int = torch.clamp((mantissa_scaled + 0.5).int(), 0, 3)

    quantized = torch.pow(2.0, exponent - 7) * (1.0 + mantissa_int.float() / 4.0)
    quantized[underflow] = 0.0
    quantized[overflow] = 60.0
    return quantized

def quantize_small_to_ufp6_fp32(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    mask1 = x >= 0.1
    mask2 = (x >= 0.01) & (x < 0.1)
    mask3 = (x >= 0.001) & (x < 0.01)
    mask4 = (x < 0.001)

    out[mask1] = x[mask1]
    out[mask2] = quantize_fp32_to_fp16_fp32(x[mask2])
    out[mask3] = quantize_fp32_to_fp8_fp32(x[mask3])
    out[mask4] = quantize_fp32_to_ufp6_fp32(x[mask4])
    # Else remains 0
    return out
