"""Individual-based (agent-based) bacterial model coupled to the same chip/tube fields.

Replaces the continuum biomass field X of ``model.py`` with discrete agents while
keeping the substrate (C) and temperature (T) PDEs, the supply tube, and the
interface coupling *exactly* as they were — the IBM swap is local to the biomass
representation.

Agent state (fixed pool of ``n_max`` slots, one tensor per quantity):
    pos  : (n_max, 2) positions, x in [0, 1], y in [0, ly]
    mass : (n_max,)   biomass mass of the individual (mDEB-lite: one lumped state;
                      a full mDEB reserve/structure vector would simply add columns)
    alive: (n_max,)   bool mask; dead slots (mass == 0) are reused by division

Per-step agent lifecycle (see ``agent_step``):
    1. sample C, T at the agent position (differentiable bilinear gather)
    2. mass-proportional uptake: consumption rate = mu(C_i, T_i) * m_i / Y_XC
    3. splat consumption (substrate sink) and heat (Q = Y_QC * consumption) onto the
       grid with bilinear weights -> these become the reaction terms of the C/T updates
    4. integrate mass: dm/dt = (mu - delta) * m
    5. division: an agent whose mass reaches ``m_div`` splits into two half-mass
       agents; the child occupies the lowest-indexed free slot
    6. death: agents decaying below ``m_die`` are removed (slot freed)
    7. motility: reflected random walk with diffusivity D_X

Differentiability notes (the design choices that make gradient-based control possible):
    * Gradients follow continuous quantities within a fixed division/death event
      sequence. Division preserves instantaneous mass, but changes future noise-driven
      trajectories; event timing and death thresholds are not differentiated.
    * Motility noise is passed in pre-generated (``noise`` argument) rather than sampled
      inside the rollout: the same noise stream is reused across optimizer iterations
      (common random numbers), turning the stochastic simulation into a fixed,
      piecewise differentiable surrogate landscape. It also makes the rollout checkpoint-safe.
    * Splatting/scattering use constant indices, so autograd follows the values only.
    * The division/death thresholds enter only through hard comparisons (zero gradient
      w.r.t. the threshold itself); if threshold sensitivity ever matters, replace the
      hard comparison with a sigmoid gate.

Scale mapping to the continuum model: total initial biomass per chip equals the
continuum integral (mean X 0.036 x area 0.5 = INIT_TOTAL_MASS), distributed over
``n0`` agents; agents divide at twice their initial mass (biological doubling), so
the mean-field growth dynamics of the population approximate the continuum X while
adding discrete-colony structure. Splatted rates are divided by the cell area so the
reaction fields carry the same areal-density units as the continuum terms.
"""

import torch

from .model import DEFAULT_T_REF, chip_laplacian, growth_rate, tube_step, validate_rollout

INIT_TOTAL_MASS = 0.018


def _bilinear_weights(pos, p):
    """Shared bilinear machinery: corner indices + weights for each agent position.

    pos: (N, 2), columns x in [0, 1], y in [0, ly]. Indices are clamped at the domain
    edges (edge-replicate behavior, consistent with the Neumann wall treatment).
    Returns (ix0, ix1, iy0, iy1, fx, fy), all usable for both gather and scatter.
    """
    gx = (pos[:, 0] * p.nx - 0.5).clamp(0.0, p.nx - 1.0)
    gy = (pos[:, 1] / p.ly * p.ny - 0.5).clamp(0.0, p.ny - 1.0)
    ix0 = gx.floor().clamp(0, p.nx - 1)
    iy0 = gy.floor().clamp(0, p.ny - 1)
    ix1 = (ix0 + 1).clamp(0, p.nx - 1)
    iy1 = (iy0 + 1).clamp(0, p.ny - 1)
    return ix0, ix1, iy0, iy1, gx - ix0, gy - iy0


def sample_field(field, pos, p):
    """Bilinear sample of a (ny, nx) field at agent positions; gradients flow into the field."""
    ix0, ix1, iy0, iy1, fx, fy = _bilinear_weights(pos, p)
    flat = field.reshape(-1)
    idx = lambda iy, ix: (iy * p.nx + ix).long()
    f00 = flat[idx(iy0, ix0)]
    f10 = flat[idx(iy0, ix1)]
    f01 = flat[idx(iy1, ix0)]
    f11 = flat[idx(iy1, ix1)]
    return ((1 - fy) * ((1 - fx) * f00 + fx * f10)
            + fy * ((1 - fx) * f01 + fx * f11))


def splat_density(values, pos, p):
    """Deposit per-agent rates onto the (ny, nx) grid as an areal rate density.

    Each agent distributes its rate over the 4 nodes of its containing cell (a 2dx-wide
    tent kernel — smooth enough that point agents do not create single-cell spikes);
    dividing by the cell area converts mass/time into the concentration/time units the
    PDE reaction terms use. ``scatter_add`` with constant indices keeps autograd on
    ``values``. Returns shape (ny, nx).
    """
    ix0, ix1, iy0, iy1, fx, fy = _bilinear_weights(pos, p)
    flat = torch.zeros(p.ny * p.nx, dtype=values.dtype)
    flat = flat.scatter_add(0, (iy0 * p.nx + ix0).long(), (1 - fy) * (1 - fx) * values)
    flat = flat.scatter_add(0, (iy0 * p.nx + ix1).long(), (1 - fy) * fx * values)
    flat = flat.scatter_add(0, (iy1 * p.nx + ix0).long(), fy * (1 - fx) * values)
    flat = flat.scatter_add(0, (iy1 * p.nx + ix1).long(), fy * fx * values)
    return flat.reshape(p.ny, p.nx) / (p.dx * p.dy)


def reflect(pos, p):
    """Reflect agent positions back into the chip domain (no-flux walls for motility)."""
    x = torch.remainder(pos[:, 0], 2.0)
    x = torch.where(x > 1.0, 2.0 - x, x)
    y = torch.remainder(pos[:, 1], 2.0 * p.ly)
    y = torch.where(y > p.ly, 2.0 * p.ly - y, y)
    return torch.stack([x, y], dim=1)


def init_agents(chips, p, n_max, n0, div_factor):
    """Seed each chip with n0 agents carrying the continuum-equivalent total biomass.

    The spatial composition mirrors the continuum initial condition: ~55% of the
    agents are spread uniformly over the chip (the continuum's uniform X background,
    which hugs the substrate source and grows fast), ~45% form a Gaussian cluster at
    mid-depth (the continuum blob). Matching this composition matters: a purely
    clustered seeding starves in its own depletion zone and the population stalls.
    Unused slots are dead (mass 0) and available for division.
    Returns (pos, mass, alive, m_div, m_die):
    pos (M, n_max, 2), mass (M, n_max), alive (M, n_max) bool.
    """
    if not 0 < n0 <= n_max or div_factor <= 1:
        raise ValueError("require 0 < n0 <= n_max and div_factor > 1")
    g = torch.Generator().manual_seed(1234)
    m0 = INIT_TOTAL_MASS / n0
    m_div = div_factor * m0
    m_die = 0.25 * m0
    n_blob = n0 // 2
    n_bg = n0 - n_blob
    pos_blob = torch.empty(chips, n_blob, 2).normal_(0.0, 0.06, generator=g)
    pos_blob[..., 0] += 0.5
    pos_blob[..., 1] += p.ly / 2
    pos_bg = torch.rand(chips, n_bg, 2, generator=g)
    pos_bg[..., 1] *= p.ly
    pos = torch.cat([pos_blob, pos_bg], dim=1)
    pos[..., 0] = pos[..., 0].clamp(0.0, 1.0)
    pos[..., 1] = pos[..., 1].clamp(0.0, p.ly)
    pos = torch.cat([pos, torch.zeros(chips, n_max - n0, 2)], dim=1)
    mass = torch.zeros(chips, n_max)
    mass[:, :n0] = m0
    alive = torch.zeros(chips, n_max, dtype=torch.bool)
    alive[:, :n0] = True
    return pos, mass, alive, m_div, m_die


def limit_agent_uptake(rates, pos, available, p, dt):
    """Limit each agent by the most depleted cell in its deposition footprint.

    Cell factors compare aggregate demand to post-transport substrate. Taking
    the minimum factor over nonzero bilinear weights guarantees that resplatting
    the limited agent rates cannot exhaust any cell. The same accepted rates
    drive agent growth, substrate removal, and heat production.
    """
    demand = splat_density(rates, pos, p)
    available = available.clamp_min(0)
    requested = dt * demand
    scarce = requested > available
    # Avoid division by zero even in the inactive autograd branch.
    denominator = torch.where(scarce, requested, torch.ones_like(requested))
    factors = torch.where(scarce, available.clamp_min(0) / denominator, torch.ones_like(requested))
    ix0, ix1, iy0, iy1, fx, fy = _bilinear_weights(pos, p)
    limits = []
    for ix, iy, weight in (
        (ix0, iy0, (1 - fx) * (1 - fy)),
        (ix1, iy0, fx * (1 - fy)),
        (ix0, iy1, (1 - fx) * fy),
        (ix1, iy1, fx * fy),
    ):
        local = factors[iy.long(), ix.long()]
        limits.append(torch.where(weight > 0, local, torch.ones_like(local)))
    return rates * torch.stack(limits).amin(dim=0)


def agent_step(c_field, t_field, pos, mass, alive, noise, p, dt, m_div, m_die,
               substrate_available=None):
    """One lifecycle pass of the agent pool against one chip's fields.

    Parameters
    ----------
    c_field, t_field : (ny, nx) current chip substrate / temperature fields
    pos, mass, alive : agent state — (n_max, 2), (n_max,), (n_max,)
    noise : (n_max, 2) motility increments ~ N(0, 1) for this step (pre-generated, CRN)
    m_div, m_die : division / death mass thresholds

    Returns (pos_new, mass_new, alive_new, cons_field, heat_field), the last two being
    (ny, nx) areal rate densities ready for the C/T PDE updates. Uptake is evaluated on
    the pre-division state; children inherit the parent position before everyone moves.
    """
    ci = sample_field(c_field, pos, p)
    ti = sample_field(t_field, pos, p)
    active = alive.float()
    mu_i = growth_rate(ci, ti, p)                                 # (n_max,) mass-specific

    if substrate_available is None:
        substrate_available = c_field
    consumption_rate = limit_agent_uptake(
        mu_i * mass * active / p.y_xc, pos, substrate_available, p, dt)
    cons_field = splat_density(consumption_rate, pos, p)
    heat_field = splat_density(p.y_qc * consumption_rate, pos, p)

    mass = mass + dt * (p.y_xc * consumption_rate - p.delta * mass * active)

    alive = alive & (mass > m_die)
    mass = torch.where(alive, mass, torch.zeros_like(mass))

    div_mask = alive & (mass > m_div)
    div_idx = div_mask.nonzero(as_tuple=True)[0]
    free_idx = (~alive).nonzero(as_tuple=True)[0]
    n_div = min(div_idx.numel(), free_idx.numel())
    if n_div > 0:
        div_idx = div_idx[:n_div]
        free_idx = free_idx[:n_div]
        parent_half = mass[div_idx] / 2.0                         # grad-carrying split
        mass = mass.clone()
        mass[div_idx] = parent_half
        mass[free_idx] = parent_half                              # child inherits half
        alive = alive.clone()
        alive[free_idx] = True                                    # child is alive (else the
        pos = pos.index_copy(0, free_idx, pos[div_idx])           # next step would zero its mass)

    pos = reflect(pos + noise * (2.0 * p.d_x * dt) ** 0.5, p)
    return pos, mass, alive, cons_field, heat_field


def chip_step_ibm(c, t, c_bc, t_bc, cons_field, heat_field, p, dt):
    """C/T PDE update with agent-sourced reaction terms (the X field is gone).

    Batched over chips: c, t (M, ny, nx); c_bc, t_bc, and the returned fluxes
    (M, nx); cons_field, heat_field (M, ny, nx). Identical structure to the
    continuum chip_step except the substrate sink and heat source are the
    agent-splatted rate fields instead of mu(C,T)*X expressions. The interface
    fluxes J_C, J_Q use the same ghost-gradient construction, so the tube
    coupling code is shared unchanged with ``model.tube_step``.
    """
    j_c = -2 * p.d_c * (c[..., 0, :] - c_bc) / p.dy
    j_q = -2 * p.alpha_chip * (t[..., 0, :] - t_bc) / p.dy
    # cons_field must come from availability-limited agent uptake.
    available = c + dt * p.d_c * chip_laplacian(c, c_bc, p.dx, p.dy)
    c_new = (available - dt * cons_field).clamp_min(0)  # remove roundoff after bounded uptake
    t_new = t + dt * (p.alpha_chip * chip_laplacian(t, t_bc, p.dx, p.dy) + heat_field)
    return c_new, t_new, j_c, j_q


def project_agents(mass, alive, pos, p):
    """Project agent masses onto a biomass-density field comparable to continuum X.

    Bilinear splat of alive agents' mass divided by cell area (dx*dy), so field means
    and integrals match the continuum convention used by the objective and figures.
    Accepts a single chip's arrays (mass (n_max,)) or a batch (mass (M, n_max), etc.).
    """
    if mass.dim() == 1:
        return splat_density(torch.where(alive, mass, torch.zeros_like(mass)), pos, p)
    return torch.stack([
        splat_density(torch.where(alive[m], mass[m], torch.zeros_like(mass[m])), pos[m], p)
        for m in range(mass.shape[0])
    ])


def simulate_ibm(u_steps, c_in_steps, p, noise, chips=1, n_steps=1000, dt=1.2e-3,
                 t_in=0.0, c_tube_init=1.0, t_ref=DEFAULT_T_REF, n_max=4096, n0=256,
                 div_factor=2.0, record_every=1):
    """Differentiable IBM rollout; drop-in replacement for ``model.simulate``.

    Parameters
    ----------
    u_steps, c_in_steps : (n_steps,) control values, as in the continuum model
    noise : (n_steps, n_max, 2) pre-generated motility increments (common random
        numbers — generated once with a fixed seed in run.py and reused every
        optimizer iteration)
    n_max, n0, div_factor : agent pool size, initial agents per chip, division mass factor

    Returns the same dict contract as ``model.simulate``: chip fields c/x/t (x is the
    agent-projected biomass density, with grad), tube fields, scalar temp_penalty
    (with grad), detached per-step records (now including ``n_agents``), and the final
    agent state (mass/pos/alive) for inspection.
    """
    validate_rollout(u_steps, c_in_steps, p, chips, n_steps, dt, record_every, c_tube_init)
    if noise.shape != (n_steps, n_max, 2) or not bool(torch.isfinite(noise).all()):
        raise ValueError("noise must be finite with shape (n_steps, n_max, 2)")
    c = torch.full((chips, p.ny, p.nx), 0.05)
    t = torch.zeros(chips, p.ny, p.nx)
    c_tube = torch.full((chips * p.nx,), float(c_tube_init))
    t_tube = torch.zeros(chips * p.nx)
    pos, mass, alive, m_div, m_die = init_agents(chips, p, n_max, n0, div_factor)
    tp = torch.zeros((), dtype=c.dtype)
    records = {"time": [], "biomass": [], "temp": [], "temp_max": [], "substrate": [], "tube_out": [],
               "j_c_total": [], "n_agents": []}
    peak_temp = t.amax().detach()
    for i in range(n_steps):
        c_bc = c_tube.reshape(chips, p.nx)
        t_bc = t_tube.reshape(chips, p.nx)
        available = c + dt * p.d_c * chip_laplacian(c, c_bc, p.dx, p.dy)
        cons_list, heat_list = [], []
        new_pos, new_mass, new_alive = [], [], []
        for m in range(chips):
            pos_m, mass_m, alive_m, cons_m, heat_m = agent_step(
                c[m], t[m], pos[m], mass[m], alive[m], noise[i], p, dt, m_div, m_die,
                substrate_available=available[m])
            new_pos.append(pos_m)
            new_mass.append(mass_m)
            new_alive.append(alive_m)
            cons_list.append(cons_m)
            heat_list.append(heat_m)
        pos, mass, alive = torch.stack(new_pos), torch.stack(new_mass), torch.stack(new_alive)
        cons, heat = torch.stack(cons_list), torch.stack(heat_list)
        c, t, j_c, j_q = chip_step_ibm(c, t, c_bc, t_bc, cons, heat, p, dt)
        c_tube, t_tube = tube_step(c_tube, t_tube, j_c, j_q, u_steps[i], c_in_steps[i],
                                   torch.tensor(float(t_in), dtype=c.dtype), p, p.dx, dt)
        tp = tp + dt * torch.clamp(t.amax() - t_ref, min=0.0).pow(2)
        peak_temp = torch.maximum(peak_temp, t.amax().detach())
        if i % record_every == 0 or i == n_steps - 1:
            # detached scalar snapshots for plotting (no graph cost)
            records["time"].append((i + 1) * dt)
            records["biomass"].append(float(project_agents(mass, alive, pos, p).mean().detach()))
            records["temp"].append(float(t.mean().detach()))
            records["temp_max"].append(float(t.amax().detach()))
            records["substrate"].append(float(c.mean().detach()))
            records["tube_out"].append(float(c_tube[-1].detach()))
            records["j_c_total"].append(float(j_c.sum().detach()) * p.dx)
            records["n_agents"].append(int(alive.sum()))
    x = torch.stack([project_agents(mass[m], alive[m], pos[m], p) for m in range(chips)])
    return {
        "c": c,
        "x": x,
        "t": t,
        "c_tube": c_tube,
        "t_tube": t_tube,
        "temp_penalty": tp,
        "peak_temp": float(peak_temp),
        "records": records,
        "agent_mass": mass,
        "agent_pos": pos,
        "agent_alive": alive,
    }
