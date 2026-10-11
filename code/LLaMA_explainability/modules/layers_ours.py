"""
LRP primitive layers for LLaMA.
Reuses BERT LRP primitives (Linear α-β rule, Add, Clone, MatMul, Mul, etc.)
and adds RMSNorm / SiLU for LLaMA-specific components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True
    self.Y = output


class RelProp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R


class RelPropSimple(RelProp):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        if not torch.is_tensor(self.X):
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * C[0]
        return outputs


# ── Reused from BERT ──────────────────────────────────────────────

class Softmax(nn.Softmax, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class MatMul(RelPropSimple):
    def forward(self, inputs):
        return torch.matmul(*inputs)


class Mul(RelPropSimple):
    def forward(self, inputs):
        return torch.mul(*inputs)


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        return [a, b]


class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        return [input for _ in range(num)]

    def relprop(self, R, alpha):
        Z = [self.X for _ in range(self.num)]
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]
        return self.X * C


class IndexSelect(RelProp):
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)
        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        if not torch.is_tensor(self.X):
            outputs = [self.X[0] * C[0], self.X[1] * C[1]]
        else:
            outputs = self.X * C[0]
        return outputs


class Sequential(nn.Sequential):
    def relprop(self, R, alpha):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R


class Linear(nn.Linear, RelProp):
    """LRP α-β rule for linear layers."""
    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            C2 = x2 * self.gradprop(Z2, x2, S2)[0]
            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)
        return alpha * activator_relevances - beta * inhibitor_relevances


# ── New for LLaMA ─────────────────────────────────────────────────

class RMSNorm(RelProp):
    """LLaMA RMSNorm with identity LRP rule (norm layers don't change relevance)."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


class SiLU(RelProp):
    """SiLU activation with identity LRP rule."""
    def forward(self, x):
        return F.silu(x)
