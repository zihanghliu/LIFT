import torch
import math

from torch import nn
from torch.nn.modules import Module
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


class S2ColumnLinear(Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        start=None,
        end=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=True,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(out_features, **factory_kwargs), requires_grad=True
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.start = start
        self.end = end

        self.s2 = nn.Parameter(
            torch.zeros(end - start, in_features), requires_grad=True
        )
        self.weight.requires_grad = False
        self.fused = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def fuse_s2_weight(self):
        if self.fused == True:
            return
        self.weight.data[self.start : self.end, :] += self.s2
        self.fused = True

    def unfuse_s2_weight(self):
        if self.fused == False:
            return
        self.weight[self.start : self.end, :] -= self.s2
        self.fused = False

    def forward(self, input: Tensor) -> Tensor:
        base_output = torch.nn.functional.linear(input, self.weight, self.bias)
        if self.fused:
            return base_output
        else:
            s2_output = torch.nn.functional.linear(input, self.s2, None)
            base_output[:, :, self.start : self.end] += s2_output
            return base_output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class S2RowLinear(Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        start=None,
        end=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=True,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(out_features, **factory_kwargs), requires_grad=True
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.start = start
        self.end = end

        self.s2 = nn.Parameter(
            torch.zeros(out_features, end - start), requires_grad=True
        )
        self.weight.requires_grad = False
        self.fused = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def fuse_s2_weight(self):
        if self.fused == True:
            return
        self.weight.data[:, self.start : self.end] += self.s2
        self.fused = True

    def unfuse_s2_weight(self):
        if self.fused == False:
            return
        self.weight[:, self.start : self.end] -= self.s2
        self.fused = False

    def forward(self, input: Tensor) -> Tensor:
        base_output = torch.nn.functional.linear(input, self.weight, self.bias)
        if self.fused:
            return base_output
        else:
            s2_output = torch.nn.functional.linear(
                input[:, :, self.start : self.end], self.s2, None
            )
            base_output += s2_output
            return base_output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
