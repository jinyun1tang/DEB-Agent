"""Differentiable forward model of the coupled 1D-tube / 2D-chip microfluidic system.

State layout
------------
Chip fields ``c`` (substrate), ``x`` (biomass), ``t`` (temperature deviation from the
thermal optimum) are dense torch tensors of shape ``(M, ny, nx)`` — chip index ``m``
is a pure batch dimension, so M chips advance in lockstep with no Python loop.

Tube fields ``c_tube``, ``t_tube`` are 1D tensors of shape ``(M * nx,)``. Chip ``m``
owns the contiguous tube cells ``[m*nx, (m+1)*nx)``; this tiling is what makes the
1D<->2D coupling two plain ``reshape`` calls (see ``simulate``).

Units are nondimensional. Temperature is measured as a *deviation from the growth
optimum*, so ``T = 0`` is ideal, ``T > 0`` suppresses growth via ``growth_rate``.

Sign conventions for the interface fluxes (both defined positive *into the chip*):
    J_C = -D_C dC/dy |_{y=0}  > 0  means substrate leaves the tube and feeds the chip
    J_Q = -alpha dT/dy |_{y=0}< 0  means the chip is hotter than the tube and sheds heat

Everything here is built from differentiable torch primitives (cat / slicing / exp /
clamp), so autodiff flows through arbitrary-length rollouts — that property is the
entire point of the module; it is what allows gradient-based optimal control in
``control.py`` / ``run.py`` without a hand-derived adjoint model.
"""

import torch
from dataclasses import dataclass

DEFAULT_T_REF = 0.10


@dataclass
class Params:
    """Nondimensional physical and numerical parameters (see design doc, section 1)."""

    # --- chip grid: domain [0, 1] x [0, ly], nx x ny cell-centered nodes ---
    nx: int = 32
    ny: int = 16
    dx: float = 1.0 / 32
    ly: float = 0.5
    dy: float = 0.5 / 16

    # --- transport coefficients ---
    d_c: float = 0.05      # substrate diffusivity (chip and tube)
    d_x: float = 0.01      # biomass random-motility diffusivity (chip only)
    alpha_chip: float = 0.1    # chip thermal diffusivity; > d_c so heat relaxes faster
    alpha_tube: float = 0.04   # tube thermal diffusivity

    # --- Monod kinetics with Gaussian (Cardinal-type) thermal scaling ---
    mu0: float = 1.5           # max growth rate at T = 0 (the optimum)
    thermal_width: float = 0.45  # width of the thermal choke: mu_max(T) = mu0*exp(-(T/w)^2)
    ks: float = 0.10           # Monod half-saturation constant
    y_xc: float = 0.5          # biomass yield per unit substrate consumed
    y_qc: float = 1.0          # metabolic heat released per unit substrate consumed
    delta: float = 0.08        # biomass decay rate

    h_tube: float = 0.1        # tube height; divides interface fluxes -> 1/H amplifies coupling

    def max_stable_dt(self, u_max=1.5):
        """Conservative positivity bound including half-cell boundaries and coupling.

        The chip boundary diagonal has at most 2/dx² + 3/dy²; the tube
        includes upwind advection, inlet diffusion, and interface exchange.
        Biomass diffusion and explicit decay are included separately.
        """
        chip_rate = max(self.d_c, self.alpha_chip) * (2 / self.dx**2 + 3 / self.dy**2)
        biomass_rate = 2 * self.d_x * (1 / self.dx**2 + 1 / self.dy**2) + self.delta
        tube_rate = max(self.d_c, self.alpha_tube) * 3 / self.dx**2 + u_max / self.dx
        tube_rate += 2 * max(self.d_c, self.alpha_chip) / (self.dy * self.h_tube)
        return 0.9 / max(chip_rate, biomass_rate, tube_rate)


def growth_rate(c, t, p):
    """Local growth rate mu(C, T) = mu_max(T) * C / (Ks + C).

    mu_max(T) uses a smooth Gaussian cardinal model in the temperature *deviation*,
    which is differentiable everywhere (unlike a hard cutoff) — important because
    the optimal-control gradients must pass through it.
    """
    mu_max = p.mu0 * torch.exp(-((t / p.thermal_width) ** 2))
    return mu_max * c / (p.ks + c)


def neumann_laplacian_2d(f, dx, dy):
    """Central-difference Laplacian with zero-flux (Neumann) walls on all four sides.

    Ghost cells are made by edge replication: a replicated neighbor yields
    (f[i+1] - f[i]) / dx at the boundary face, i.e. exactly zero normal gradient.
    """
    fx = torch.cat([f[..., :1], f, f[..., -1:]], dim=-1)
    d2x = (fx[..., 2:] - 2 * fx[..., 1:-1] + fx[..., :-2]) / dx**2
    fy = torch.cat([f[..., :1, :], f, f[..., -1:, :]], dim=-2)
    d2y = (fy[..., 2:, :] - 2 * fy[..., 1:-1, :] + fy[..., :-2, :]) / dy**2
    return d2x + d2y


def chip_laplacian(f, bottom, dx, dy):
    """Central-difference Laplacian with the problem's mixed chip boundary conditions.

    * y = 0 (interface): Dirichlet — ``bottom`` holds the tube values for each chip,
      shape (M, nx), half a cell away from row 0.
    * y = ly and both x walls: zero-flux (edge replication, as above).

    The ghost value 2*bottom - f[0] places the prescribed value at the face.
    The matching interface flux is 2*D*(bottom - f[0])/dy.
    """
    fx = torch.cat([f[..., :1], f, f[..., -1:]], dim=-1)
    d2x = (fx[..., 2:] - 2 * fx[..., 1:-1] + fx[..., :-2]) / dx**2
    fy = torch.cat([2 * bottom.unsqueeze(-2) - f[..., :1, :], f, f[..., -1:, :]], dim=-2)
    d2y = (fy[..., 2:, :] - 2 * fy[..., 1:-1, :] + fy[..., :-2, :]) / dy**2
    return d2x + d2y


def tube_laplacian(f, inlet, dx):
    """1D central-difference Laplacian with Dirichlet inlet / zero-gradient outlet.

    The inlet is half a cell from cell 0: ghost = 2*inlet - f[0].
    The right ghost replicates the last cell.
    """
    padded = torch.cat([2 * inlet.reshape((1,) * f.dim()) - f[..., :1], f, f[..., -1:]], dim=-1)
    return (padded[..., 2:] - 2 * padded[..., 1:-1] + padded[..., :-2]) / dx**2


def chip_step(c, x, t, c_bc, t_bc, p, dt):
    """Advance chip fields one explicit-Euler step; return updated fields + interface fluxes.

    Parameters
    ----------
    c, x, t : (M, ny, nx) substrate, biomass, temperature-deviation
    c_bc, t_bc : (M, nx) tube values applied as Dirichlet boundary at y = 0
    dt : time step

    Returns
    -------
    c_new, x_new, t_new : updated chip fields
    j_c, j_q : (M, nx) interface fluxes INTO the chip (sign convention in module docstring)

    Notes
    -----
    * ``consumption = mu * X / Y_{X/C}`` is the substrate sink R_C; the same quantity
      times y_qc is the volumetric heat source Q_bio — one nonlinear term drives both.
    * Uptake is limited by substrate remaining after transport; the same limited
      rate drives biomass gain and metabolic heat. T is a signed deviation.
    * The fluxes are evaluated *before* the update, from the pre-step boundary-row
      gradient, and are exactly the ghost gradient used inside chip_laplacian — this is
      what makes the Dirichlet/Neumann coupling pair consistent.
    """
    mu = growth_rate(c, t, p)
    available = c + dt * p.d_c * chip_laplacian(c, c_bc, p.dx, p.dy)
    consumption = torch.minimum(mu * x / p.y_xc, available.clamp_min(0) / dt)
    j_c = -2 * p.d_c * (c[..., 0, :] - c_bc) / p.dy
    j_q = -2 * p.alpha_chip * (t[..., 0, :] - t_bc) / p.dy
    c_new = (available - dt * consumption).clamp_min(0)  # roundoff only: uptake was bounded above
    x_new = x + dt * (p.d_x * neumann_laplacian_2d(x, p.dx, p.dy) + p.y_xc * consumption - p.delta * x)
    t_new = t + dt * (p.alpha_chip * chip_laplacian(t, t_bc, p.dx, p.dy) + p.y_qc * consumption)
    return c_new, x_new, t_new, j_c, j_q


def tube_step(c_tube, t_tube, j_c, j_q, u, c_in, t_in, p, dx_tube, dt):
    """Advance the 1D tube one explicit-Euler step (advection + diffusion + interface sinks).

    Parameters
    ----------
    c_tube, t_tube : (M*nx,) tube fields; cell m*nx..(m+1)*nx-1 couples to chip m
    j_c, j_q : (M, nx) fluxes into each chip, from ``chip_step``
    u, c_in, t_in : scalar (0-dim tensor) controls — flow velocity, inlet substrate, inlet temperature

    The tube equation of the problem statement, dC/dt + u dC/dx = D d2C/dx2 - J/H, is
    discretized as: first-order upwind for advection (ghost at the inlet carries the
    inflow value, so cell 0 feels C_in immediately), central diffusion, and the
    interface flux as a distributed sink ``-J / H_tube``.

    The single ``reshape(M*nx)`` maps each chip's (nx,) flux row onto its contiguous
    tube segment — this line is the whole multi-chip coupling. Sign check: a positive
    J_C (substrate flowing into the chip) appears here as ``-sink_c``, i.e. the tube
    loses exactly what the chip gains.
    """
    mt = c_tube.shape[-1]
    c_up = torch.cat([c_in.reshape((1,) * c_tube.dim()), c_tube[..., :-1]], dim=-1)
    t_up = torch.cat([t_in.reshape((1,) * t_tube.dim()), t_tube[..., :-1]], dim=-1)
    adv_c = u * (c_tube - c_up) / dx_tube
    adv_t = u * (t_tube - t_up) / dx_tube
    sink_c = j_c.reshape(mt) / p.h_tube
    sink_q = j_q.reshape(mt) / p.h_tube
    c_new = c_tube + dt * (p.d_c * tube_laplacian(c_tube, c_in, dx_tube) - adv_c - sink_c)
    t_new = t_tube + dt * (p.alpha_tube * tube_laplacian(t_tube, t_in, dx_tube) - adv_t - sink_q)
    return c_new, t_new


def initial_state(chips, p):
    """Build the initial condition: a starved chip with one Gaussian biomass seed per chip.

    C0 = 0.05 everywhere (deliberately low: with a pre-fed chip the optimal-control
    landscape degenerates — supply barely matters and the optimizer just shuts the
    inlet; the starved start makes feeding genuinely load-bearing).

    The biomass seed is placed at mid-depth, x-centered at 0.3 + 0.4*i/(M-1) so that
    multi-chip visualizations show distinguishable colonies. A small uniform
    background (0.02) keeps growth possible everywhere.
    """
    yy = torch.linspace(0.5 * p.dy, p.ly - 0.5 * p.dy, p.ny).view(1, p.ny, 1)
    xx = torch.linspace(0.5 / p.nx, 1.0 - 0.5 / p.nx, p.nx).view(1, 1, p.nx)
    centers = torch.tensor([0.3 + 0.4 * i / max(chips - 1, 1) for i in range(chips)]).view(chips, 1, 1)
    blob = torch.exp(-(((xx - centers) ** 2 + (yy - p.ly / 2) ** 2) / 0.01))
    c0 = torch.full((chips, p.ny, p.nx), 0.05)
    x0 = 0.02 + 0.25 * blob
    t0 = torch.zeros(chips, p.ny, p.nx)
    return c0, x0, t0


def validate_rollout(u_steps, c_in_steps, p, chips, n_steps, dt, record_every, c_tube_init):
    """Reject invalid controls or unstable transport before building a rollout tape."""
    if chips < 1 or n_steps < 1 or record_every < 1 or dt <= 0:
        raise ValueError("chips, n_steps, record_every and dt must be positive")
    if u_steps.shape != (n_steps,) or c_in_steps.shape != (n_steps,):
        raise ValueError("controls must have shape (n_steps,)")
    if not bool(torch.isfinite(u_steps).all() & torch.isfinite(c_in_steps).all()):
        raise ValueError("controls must be finite")
    if bool((u_steps < 0).any() | (c_in_steps < 0).any()) or c_tube_init < 0:
        raise ValueError("flow and substrate must be nonnegative")
    limit = p.max_stable_dt(float(u_steps.detach().max()))
    if dt > limit:
        raise ValueError(f"dt={dt} exceeds transport/decay stability limit {limit:.6g}")


def simulate(u_steps, c_in_steps, p, chips=1, n_steps=800, dt=1.5e-3, t_in=0.0, c_tube_init=1.0, t_ref=DEFAULT_T_REF, record_every=1):
    """Differentiable rollout of the coupled system; the objective reads its outputs.

    Parameters
    ----------
    u_steps, c_in_steps : (n_steps,) piecewise-constant control values, one per step
        (expanded from the K-segment parameterization by ``PiecewiseControl.step_values``)
    t_ref : overheating threshold; the returned ``temp_penalty`` accumulates
        dt * [max_chip(T) - t_ref]^2 each step — penalizing the spatial *maximum*
        (the hotspot at the colonies) rather than the domain mean, because it is the
        hotspot that biologically matters.

    Returns
    -------
    dict with final chip fields ``c/x/t`` (with grad, for the loss), final tube fields,
    the scalar ``temp_penalty`` (with grad), and ``records`` — detached per-step scalars
    (mean biomass, mean T, mean C, tube outlet concentration, total interface substrate
    flux) for plotting. Records are floats, so they carry no graph cost.
    """
    validate_rollout(u_steps, c_in_steps, p, chips, n_steps, dt, record_every, c_tube_init)
    c, x, t = initial_state(chips, p)
    c_tube = torch.full((chips * p.nx,), float(c_tube_init))
    t_tube = torch.zeros(chips * p.nx)
    tp = torch.zeros((), dtype=c.dtype)
    records = {"time": [], "biomass": [], "temp": [], "temp_max": [], "substrate": [], "tube_out": [], "j_c_total": []}
    peak_temp = t.amax().detach()
    for i in range(n_steps):
        u = u_steps[i]
        c_in = c_in_steps[i]
        # tube values at each chip's interface segment -> Dirichlet boundary rows
        c_bc = c_tube.reshape(chips, p.nx)
        t_bc = t_tube.reshape(chips, p.nx)
        c, x, t, j_c, j_q = chip_step(c, x, t, c_bc, t_bc, p, dt)
        c_tube, t_tube = tube_step(c_tube, t_tube, j_c, j_q, u, c_in, torch.tensor(float(t_in), dtype=c.dtype), p, p.dx, dt)
        tp = tp + dt * torch.clamp(t.amax() - t_ref, min=0.0).pow(2)
        peak_temp = torch.maximum(peak_temp, t.amax().detach())
        if i % record_every == 0 or i == n_steps - 1:
            records["time"].append((i + 1) * dt)
            records["biomass"].append(float(x.mean().detach()))
            records["temp"].append(float(t.mean().detach()))
            records["temp_max"].append(float(t.amax().detach()))
            records["substrate"].append(float(c.mean().detach()))
            records["tube_out"].append(float(c_tube[-1].detach()))
            records["j_c_total"].append(float(j_c.sum().detach()) * p.dx)
    return {
        "c": c,
        "x": x,
        "t": t,
        "c_tube": c_tube,
        "t_tube": t_tube,
        "temp_penalty": tp,
        "peak_temp": float(peak_temp),
        "records": records,
    }
