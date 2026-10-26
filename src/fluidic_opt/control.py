"""Control parameterization for gradient-based optimal control.

The controls u(t) (tube flow velocity) and C_in(t) (inlet substrate concentration)
are piecewise constant over K segments covering the horizon. Each segment holds two
raw parameters that are squashed through sigmoids onto physical bounds:

    u(t)    = u_lo + (u_hi - u_lo) * sigmoid(theta_u[k])
    C_in(t) = c_lo + (c_hi - c_lo) * sigmoid(theta_c[k])

Why sigmoid bounds instead of clamping: sigmoid keeps the parameterization smooth
and the gradient alive everywhere (a clamp kills the gradient once a bound is hit),
which matters because Adam may push controls far toward their limits during training.

Segment selection per time step is a plain integer index (floor(t / dt_seg)) —
piecewise-constant controls have zero gradient within a segment anyway, so the
non-differentiable "which segment" choice loses nothing: all gradient flows into the
selected segment's parameters.
"""

import math
import torch


class PiecewiseControl(torch.nn.Module):
    """K-segment piecewise-constant (u, C_in) schedule as trainable nn.Parameters.

    ``u0``/``c0`` set the *initial* operating point, expressed through the same
    sigmoid mapping so that the untrained parameterization reproduces the baseline
    constant control exactly (logit-inverse initialization).
    """

    def __init__(self, segments, u_bounds=(0.05, 1.5), c_bounds=(0.0, 2.0), u0=0.5, c0=1.0):
        super().__init__()
        if segments < 1:
            raise ValueError("segments must be positive")
        self.segments = segments
        self.u_bounds = u_bounds
        self.c_bounds = c_bounds
        # invert sigmoid to place the raw parameters exactly at the baseline point
        u_mid = u0 - u_bounds[0]
        u_raw0 = math.log(u_mid / (u_bounds[1] - u_bounds[0] - u_mid)) if 0 < u_mid < u_bounds[1] - u_bounds[0] else 0.0
        c_mid = c0 - c_bounds[0]
        c_raw0 = math.log(c_mid / (c_bounds[1] - c_bounds[0] - c_mid)) if 0 < c_mid < c_bounds[1] - c_bounds[0] else 0.0
        self.u_raw = torch.nn.Parameter(torch.full((segments,), u_raw0))
        self.c_raw = torch.nn.Parameter(torch.full((segments,), c_raw0))

    def u(self):
        """Bounded flow velocity per segment, shape (K,)."""
        lo, hi = self.u_bounds
        return lo + (hi - lo) * torch.sigmoid(self.u_raw)

    def c_in(self):
        """Bounded inlet substrate concentration per segment, shape (K,)."""
        lo, hi = self.c_bounds
        return lo + (hi - lo) * torch.sigmoid(self.c_raw)

    def step_values(self, n_steps, dt, t_end):
        """Expand segment values to per-step arrays for ``simulate``.

        Sample each segment at the start of the Euler step. Segment lengths differ
        by at most one step; costs must integrate these exact sampled controls.
        """
        if n_steps < self.segments or dt <= 0 or t_end <= 0:
            raise ValueError("require n_steps >= segments and positive dt and t_end")
        if not math.isclose(n_steps * dt, t_end, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("n_steps * dt must equal t_end")
        # Integer arithmetic avoids floating-point errors at segment boundaries.
        index = torch.arange(n_steps, device=self.u_raw.device) * self.segments // n_steps
        return self.u()[index], self.c_in()[index]
