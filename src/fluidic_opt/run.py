"""Training driver: gradient-based optimal control of the coupled microfluidic system.

Pipeline
--------
1. Parameterize (u(t), C_in(t)) as K sigmoid-bounded piecewise-constant segments
   (``control.PiecewiseControl``), initialized at the constant baseline (0.5, 1.0).
2. Differentiably roll out the coupled 1D/2D physics for the full horizon
   (``model.simulate``) and evaluate the objective below.
3. Backpropagate through the entire rollout and update the segment parameters with Adam.
4. Re-evaluate final and baseline controls without grad, emit figures + npz artifacts.

Objective (all nondimensional):

    J = -W_X * sum_chips mean_space(X(t_end))                       biomass reward
        + W_C * sum_n u_n C_in_n dt                                  substrate pumped
        + W_U * sum_n u_n^2 dt                                       pumping effort
        + W_T * sum_n [max_chip T_n - T_REF]_+^2 dt                  hotspot overheating

The biomass reward SUMS over chips (a shared tube feeds M chips at the price of one
inlet stream): using the plain mean here made feeding look M times less profitable and
drove the multi-chip optimizer into starvation — see design doc section 5.

Weight values were not picked blindly: degenerate optima under earlier settings are
documented in the design doc (section 3). The diagnostic that settled them is a sweep
of constant controls — check that J has an interior optimum before spending iterations.

Run from ``src/``:
    python -m fluidic_opt.run --chips 1 --iters 80 --steps 2084 --dt 1.2e-3 --t-end 2.5
"""

import argparse
from dataclasses import asdict
import json
import math
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

from fluidic_opt.control import PiecewiseControl
from fluidic_opt.model import DEFAULT_T_REF, Params, simulate
from fluidic_opt.model_ibm import simulate_ibm

# --- objective weights (see module docstring) ---
W_X = 4.0    # value of final biomass
W_C = 0.05   # price of pumped substrate: low enough that feeding can be profitable,
             # high enough that wasting it (washing through unconsumed) is punished
W_U = 0.02   # mild quadratic pumping effort, mostly for smoothness
W_T = 3.0    # overheating penalty on the chip-wide temperature maximum
T_REF = DEFAULT_T_REF # shared by the objective, both solvers, and reporting


def evaluate(ctrl, sim_fn, n_steps, dt, t_end, grad=True):
    """One full rollout + objective evaluation.

    ``sim_fn(u_steps, c_steps) -> simulate-output`` abstracts the biomass model:
    continuum (``model.simulate``) or individual-based (``model_ibm.simulate_ibm``).

    With ``grad=True`` the call builds the tape through all n_steps Euler steps
    (memory: small state tensors x n_steps — fine at this grid size); the Adam loop
    calls it with grad, the final reporting with grad=False.

    Returns (loss, simulate-output dict, substrate_cost, flow_cost).
    """
    with torch.set_grad_enabled(grad):
        u_steps, c_steps = ctrl.step_values(n_steps, dt, t_end)
        out = sim_fn(u_steps, c_steps)
        substrate_cost = (u_steps * c_steps).sum() * dt
        flow_cost = (u_steps ** 2).sum() * dt
        biomass = out["x"].sum(dim=0).mean()  # sum over chips, mean over space
        loss = -W_X * biomass + W_C * substrate_cost + W_U * flow_cost + W_T * out["temp_penalty"]
    return loss, out, substrate_cost, flow_cost


def train(ctrl, sim_fn, n_steps, dt, t_end, iters, lr):
    """Adam loop over the control parameters; returns the per-iteration loss history."""
    opt = torch.optim.Adam(ctrl.parameters(), lr=lr)
    history = []
    for it in range(iters):
        loss, out, cost_c, cost_u = evaluate(ctrl, sim_fn, n_steps, dt, t_end, grad=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            print(f"iter {it:4d}  loss {float(loss.detach()):+.5f}  biomass {float(out['x'].sum(dim=0).mean().detach()):.4f}  "
                  f"substrate_cost {float(cost_c.detach()):.4f}  temp_penalty {float(out['temp_penalty'].detach()):.5f}")
    return history


def plot_fields(out, path, chips, p):
    """Final C / X / T maps, one row of three panels per chip.

    Fields are lightly smoothed (3x3 box) for display: in the IBM mode the biomass
    projection carries single-agent speckle (~0.5 agents per cell), which would
    otherwise drown the colony structure in sampling noise.
    """
    cols = [("C (substrate)", out["c"]), ("X (biomass)", out["x"]), ("T (temp. dev.)", out["t"])]
    fig, axes = plt.subplots(chips, 3, figsize=(13, 3.2 * chips), squeeze=False)
    for m in range(chips):
        for j, (title, field) in enumerate(cols):
            ax = axes[m][j]
            arr = field.detach().numpy()[m]
            arr = uniform_filter(arr, size=3)
            im = ax.imshow(arr, origin="lower", extent=[0, 1, 0, p.ly], cmap="coolwarm", aspect="auto")
            ax.set_title(f"chip {m}: {title}", fontsize=10)
            fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def step_xy(values, t_end):
    """Expand K segment values into step-plot (x, y) arrays with where='post' edges.

    Boundaries b0..bK; xs = [b0, b1, b1, b2, ..., bK] (length 2K) pairs with
    ys = [u0, u0, u1, u1, ..., u_{K-1}, u_{K-1}] so each half-open segment
    [b_k, b_k+1) draws its own constant value.
    """
    segs = np.linspace(0.0, t_end, len(values) + 1)
    xs = np.repeat(segs, 2)[1:-1]
    ys = np.repeat(values, 2)
    return xs, ys


def plot_curves(history, baseline_out, opt_out, ctrl, t_end, path):
    """2x2 diagnostic panel: loss curve, optimized u(t) and C_in(t) vs baseline, time series."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0][0].plot(history)
    axes[0][0].set_title("loss vs iteration")
    axes[0][0].set_xlabel("iteration")

    u_seg = ctrl.u().detach().numpy()
    c_seg = ctrl.c_in().detach().numpy()
    xs, ys = step_xy(u_seg, t_end)
    axes[0][1].step(xs, ys, where="post", label="optimized")
    axes[0][1].axhline(0.5, color="grey", ls="--", label="baseline u")
    axes[0][1].set_title("flow velocity u(t)")
    axes[0][1].legend()

    xs, ys = step_xy(c_seg, t_end)
    axes[1][0].step(xs, ys, where="post", label="optimized")
    axes[1][0].axhline(1.0, color="grey", ls="--", label="baseline C_in")
    axes[1][0].set_title("inlet substrate C_in(t)")
    axes[1][0].legend()

    # time series: biomass (reward) and mean chip temperature (thermal loop) for both controls
    tt = np.array(baseline_out["records"]["time"])
    axes[1][1].plot(tt, baseline_out["records"]["biomass"], "k--", label="biomass baseline")
    axes[1][1].plot(tt, opt_out["records"]["biomass"], "g-", label="biomass optimized")
    axes[1][1].plot(tt, baseline_out["records"]["temp_max"], "r--", label="hotspot temp baseline")
    axes[1][1].plot(tt, opt_out["records"]["temp_max"], "r-", label="hotspot temp optimized")
    axes[1][1].axhline(T_REF, color="orange", ls=":", label="T penalty threshold")
    axes[1][1].set_title("mean biomass / hotspot temperature vs time")
    axes[1][1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chips", type=int, default=1, help="number of chips tiled along the shared tube")
    ap.add_argument("--iters", type=int, default=60, help="Adam iterations (full rollout + backward each)")
    ap.add_argument("--steps", type=int, default=None, help="Euler steps; otherwise derived from horizon and maximum dt")
    ap.add_argument("--t-end", type=float, default=1.2, help="horizon length (nondimensional)")
    ap.add_argument("--dt", type=float, default=1.5e-3, help="maximum time step; actual dt is t_end / steps")
    ap.add_argument("--segments", type=int, default=8, help="K piecewise-constant control segments")
    ap.add_argument("--lr", type=float, default=0.05, help="Adam learning rate on the 2K raw parameters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ibm", action="store_true", help="individual-based biomass model (model_ibm) instead of continuum X")
    ap.add_argument("--n-max", type=int, default=4096, help="IBM: agent pool size (slots, incl. free)")
    ap.add_argument("--n0", type=int, default=256, help="IBM: initial agents per chip")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "fluidic_control"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)  # float64: cheap at this size, and keeps the
    # explicit diffusion stable margins honest instead of accumulating fp32 drift over ~2k steps
    p = Params()
    if args.dt <= 0 or args.t_end <= 0 or args.chips < 1 or args.segments < 1 or args.iters < 0:
        ap.error("dt, t-end, chips and segments must be positive; iters must be nonnegative")
    max_dt = min(args.dt, p.max_stable_dt())
    if args.steps is None:
        args.steps = max(args.segments, math.ceil(args.t_end / max_dt))
    if args.steps < args.segments:
        ap.error("steps must be at least segments")
    actual_dt = args.t_end / args.steps
    if actual_dt > max_dt * (1 + 1e-12):
        ap.error(f"steps must be at least {math.ceil(args.t_end / max_dt)} for the requested horizon and dt limit")
    args.dt = actual_dt
    out_dir = os.path.join(args.out, f"{'ibm' if args.ibm else 'cont'}_chips{args.chips}")
    os.makedirs(out_dir, exist_ok=True)

    if args.ibm:
        # common random numbers: one fixed noise stream reused by every optimizer
        # iteration -> deterministic differentiable surrogate landscape (model_ibm docstring)
        g = torch.Generator().manual_seed(args.seed + 1)
        noise = torch.randn(args.steps, args.n_max, 2, generator=g)
        sim_fn = lambda u, c: simulate_ibm(u, c, p, noise, chips=args.chips,
                                           n_steps=args.steps, dt=args.dt,
                                           n_max=args.n_max, n0=args.n0, t_ref=T_REF)
    else:
        sim_fn = lambda u, c: simulate(u, c, p, chips=args.chips, n_steps=args.steps, dt=args.dt, t_ref=T_REF)

    ctrl = PiecewiseControl(args.segments)
    history = train(ctrl, sim_fn, args.steps, args.dt, args.t_end, args.iters, args.lr)

    # final re-evaluations without grad: optimized schedule vs the constant baseline it started from
    loss_opt, out_opt, cost_c, cost_u = evaluate(ctrl, sim_fn, args.steps, args.dt, args.t_end, grad=False)
    baseline = PiecewiseControl(args.segments, u0=0.5, c0=1.0)
    loss_base, out_base, _, _ = evaluate(baseline, sim_fn, args.steps, args.dt, args.t_end, grad=False)

    print("\n=== summary ===")
    print(f"loss baseline {float(loss_base):+.5f}  ->  optimized {float(loss_opt):+.5f}")
    print(f"final biomass baseline {float(out_base['x'].sum(dim=0).mean()):.4f}  ->  optimized {float(out_opt['x'].sum(dim=0).mean()):.4f}")
    print(f"substrate cost baseline {0.5 * 1.0 * args.t_end:.4f}  ->  optimized {float(cost_c):.4f}")
    print(f"peak chip temp baseline {out_base['peak_temp']:.4f}  ->  optimized {out_opt['peak_temp']:.4f} (threshold {T_REF:.2f})")
    if args.ibm:
        print(f"agent population final baseline {out_base['records']['n_agents'][-1]}  ->  optimized {out_opt['records']['n_agents'][-1]} (pool {args.n_max})")
    if args.chips > 1:
        # shared-tube competition diagnostic: mean tube concentration over each chip's segment
        ct = out_opt["c_tube"].reshape(args.chips, p.nx)
        print("per-chip mean inlet substrate (shared-tube depletion):",
              [round(float(ct[m].mean()), 4) for m in range(args.chips)])

    plot_fields(out_opt, os.path.join(out_dir, "fig_fields.png"), args.chips, p)
    plot_curves(history, out_base, out_opt, ctrl, args.t_end, os.path.join(out_dir, "fig_control.png"))
    extra = {}
    if args.ibm:
        extra = {"n_agents_base": np.array(out_base["records"]["n_agents"]),
                 "n_agents_opt": np.array(out_opt["records"]["n_agents"])}
    np.savez(
        os.path.join(out_dir, "results.npz"),
        loss_history=np.array(history),
        u_seg=ctrl.u().detach().numpy(),
        c_seg=ctrl.c_in().detach().numpy(),
        x_final=out_opt["x"].detach().numpy(),
        c_final=out_opt["c"].detach().numpy(),
        t_final=out_opt["t"].detach().numpy(),
        biomass_baseline=np.array(out_base["records"]["biomass"]),
        biomass_opt=np.array(out_opt["records"]["biomass"]),
        temp_baseline=np.array(out_base["records"]["temp"]),
        temp_opt=np.array(out_opt["records"]["temp"]),
        temp_max_baseline=np.array(out_base["records"]["temp_max"]),
        temp_max_opt=np.array(out_opt["records"]["temp_max"]),
        time=np.array(out_opt["records"]["time"]),
        metadata=json.dumps({"schema_version": 2, "params": asdict(p), "run": vars(args),
                             "t_ref": T_REF, "weights": {"biomass": W_X, "substrate": W_C,
                                                         "flow": W_U, "temperature": W_T}}),
        **extra,
    )
    print(f"artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
