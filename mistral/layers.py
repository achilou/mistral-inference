import torch
import torch.nn as nn
from EETQ import quant_weights, w8_a16_gemm

import time


def get_linear(weight, quantize):
    # if quantize == 'eetq':
    #     linear = nn.Linear(1, 1, bias=False)
    #     linear.weight = nn.Parameter(weight.to(torch.float16), requires_grad=False)
    #     linear.to('cuda')
    #     return linear
    if quantize == 'eetq':
        return EETQLinear(weight)


class EETQLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        device = weight.device
        weight = torch.t(weight).to(torch.float16).contiguous().cpu()
        start_time = time.time()
        weight, scale = quant_weights(weight, torch.int8, False)
        print(f"EETQLinear took {time.time() - start_time}")
        self.weight = weight.to('cuda')
        self.scale = scale.to('cuda')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = w8_a16_gemm(inputs, self.weight, self.scale)
        return output
