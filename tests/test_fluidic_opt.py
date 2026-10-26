"""Numerical regression checks. Run: PYTHONPATH=src python -m unittest discover -s tests."""

import unittest

import torch

from fluidic_opt.control import PiecewiseControl
from fluidic_opt.model import (
    DEFAULT_T_REF, Params, chip_laplacian, chip_step, growth_rate,
    simulate, tube_laplacian, tube_step,
)
from fluidic_opt.model_ibm import agent_step, chip_step_ibm, limit_agent_uptake, simulate_ibm
from fluidic_opt.run import evaluate


class FluidicNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls.previous_dtype)

    def setUp(self):
        self.p = Params()

    def test_face_dirichlet_matches_analytic_steady_diffusion(self):
        # D*C'' = sink, C(0)=bc, C'(ly)=0. Error should be O(dy²),
        # without the O(dy) resistance introduced by a full-cell interface.
        errors = []
        for ny in (8, 16, 32):
            dy, d, bc, sink = .5 / ny, .05, 10., 1.
            basis = torch.eye(ny).reshape(ny, ny, 1)
            operator = chip_laplacian(basis, torch.zeros(ny, 1), 1., dy).squeeze(-1).T
            boundary = chip_laplacian(torch.zeros(1, ny, 1), torch.tensor([[bc]]), 1., dy).flatten()
            numerical = torch.linalg.solve(d * operator, torch.full((ny,), sink) - d * boundary)
            y = (torch.arange(ny) + .5) * dy
            exact = bc + sink / (2 * d) * (y**2 - y)
            errors.append(float((numerical - exact).abs().max()))
        self.assertAlmostEqual(errors[0] / errors[1], 4., places=8)
        self.assertAlmostEqual(errors[1] / errors[2], 4., places=7)

    def test_tube_inlet_is_half_a_cell_away(self):
        result = tube_laplacian(torch.tensor([2., 3., 4.]), torch.tensor(1.), 1.)
        torch.testing.assert_close(result, torch.tensor([-1., 0., -1.]))

    def test_coupled_mass_and_heat_budgets(self):
        p, dt = self.p, .001
        generator = torch.Generator().manual_seed(10)
        c = .2 + .1 * torch.rand(2, p.ny, p.nx, generator=generator)
        x = .02 + .01 * torch.rand(c.shape, generator=generator)
        t = .1 * torch.rand(c.shape, generator=generator)
        ct = .3 + .1 * torch.rand(2 * p.nx, generator=generator)
        tt = .1 * torch.rand(ct.shape, generator=generator)
        u, ci, ti = torch.tensor(.5), torch.tensor(.4), torch.tensor(0.)
        cn, xn, tn, jc, jq = chip_step(c, x, t, ct.reshape(2, p.nx), tt.reshape(2, p.nx), p, dt)
        ctn, ttn = tube_step(ct, tt, jc, jq, u, ci, ti, p, p.dx, dt)
        area = p.dx * p.dy
        mass = lambda c, x, ct: (c + x / p.y_xc).sum() * area + ct.sum() * p.h_tube * p.dx
        boundary = p.h_tube * (u * (ci - ct[-1]) + 2 * p.d_c * (ci - ct[0]) / p.dx)
        expected = dt * (boundary - p.delta * x.sum() * area / p.y_xc)
        self.assertLess(abs(float(mass(cn, xn, ctn) - mass(c, x, ct) - expected)), 1e-14)
        heat = lambda t, tt: t.sum() * area + tt.sum() * p.h_tube * p.dx
        expected = dt * (p.h_tube * (u * (ti - tt[-1]) + 2 * p.alpha_tube * (ti - tt[0]) / p.dx)
                         + p.y_qc * (growth_rate(c, t, p) * x / p.y_xc).sum() * area)
        self.assertLess(abs(float(heat(tn, ttn) - heat(t, tt) - expected)), 1e-14)

    def test_continuum_scarcity_limits_growth_and_heat(self):
        p, dt = self.p, .0012
        c = torch.full((1, p.ny, p.nx), .001)
        x, t = torch.full_like(c, 100.), torch.zeros_like(c)
        cn, xn, tn, _, _ = chip_step(c, x, t, c[:, 0], t[:, 0], p, dt)
        removed = (c - cn).sum() * p.dx * p.dy
        gain = (xn - x + dt * p.delta * x).sum() * p.dx * p.dy
        torch.testing.assert_close(gain, p.y_xc * removed, atol=1e-14, rtol=1e-10)
        torch.testing.assert_close(tn.sum() * p.dx * p.dy, p.y_qc * removed)
        self.assertGreaterEqual(float(cn.min()), 0.)

    def test_agent_scarcity_preserves_uptake_growth_and_heat_budget(self):
        p, dt = self.p, .0012
        c, t = torch.full((p.ny, p.nx), .001), torch.zeros(p.ny, p.nx)
        # Uneven, overlapping footprints, including an empty cell and edge agent.
        c[7, 15] = 0
        pos = torch.tensor([[.5, .25], [.49, .24], [0., 0.]])
        mass = torch.ones(3)
        alive = torch.ones(3, dtype=torch.bool)
        available = c + dt * p.d_c * chip_laplacian(c[None], c[None, 0], p.dx, p.dy)[0]
        _, mn, _, cons, heat = agent_step(c, t, pos, mass, alive, torch.zeros_like(pos),
                                         p, dt, 10., 0., substrate_available=available)
        self.assertTrue(bool((dt * cons <= available + 1e-15).all()))
        cn, tn, _, _ = chip_step_ibm(c[None], t[None], c[None, 0], t[None, 0], cons[None], heat[None], p, dt)
        area = p.dx * p.dy
        accepted = dt * cons.sum() * area
        torch.testing.assert_close((available - cn[0]).sum() * area, accepted)
        torch.testing.assert_close((mn - mass + dt * p.delta * mass).sum(), p.y_xc * accepted)
        torch.testing.assert_close(tn.sum() * area, p.y_qc * accepted)
        self.assertGreaterEqual(float(cn.min()), 0.)

    def test_horizon_and_cost_use_exact_applied_steps(self):
        ctrl = PiecewiseControl(3)
        with torch.no_grad():
            ctrl.u_raw.copy_(torch.tensor([-1., 0., 1.]))
        u, c = ctrl.step_values(10, .1, 1.)
        torch.testing.assert_close(u, ctrl.u()[torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])])
        fake = lambda u, c: {"x": torch.zeros(1, 1, 1), "temp_penalty": torch.tensor(0.)}
        _, _, cost_c, cost_u = evaluate(ctrl, fake, 10, .1, 1.)
        torch.testing.assert_close(cost_c, (u * c).sum() * .1)
        torch.testing.assert_close(cost_u, u.square().sum() * .1)
        with self.assertRaisesRegex(ValueError, "must equal"):
            ctrl.step_values(10, .1, 2.)

    def test_limited_agent_uptake_gradient(self):
        p = self.p
        pos = torch.tensor([[.501, .251], [.49, .24]])
        rates = torch.tensor([2., 3.], requires_grad=True)
        available = torch.linspace(.001, .002, p.nx * p.ny).reshape(p.ny, p.nx)
        def objective():
            # Unequal weights exercise allocation between competing agents.
            return (limit_agent_uptake(rates, pos, available, p, .0012) * torch.tensor([1., 2.])).sum()
        objective().backward()
        ad, h = float(rates.grad[0]), 1e-5
        with torch.no_grad():
            rates[0] += h
            plus = float(objective())
            rates[0] -= 2 * h
            minus = float(objective())
            rates[0] += h
        self.assertGreater(abs(ad), 1e-6)
        self.assertAlmostEqual(ad, (plus - minus) / (2 * h), delta=1e-10)

    def test_division_preserves_mass_and_local_gradient(self):
        p = self.p
        c, t = torch.ones(p.ny, p.nx), torch.zeros(p.ny, p.nx)
        pos, mass = torch.tensor([[.5, .25], [0., 0.]]), torch.tensor([.001, 0.], requires_grad=True)
        alive = torch.tensor([True, False])
        _, mn, an, cons, _ = agent_step(c, t, pos, mass, alive, torch.zeros_like(pos),
                                       p, .0012, .0009, 0.)
        self.assertTrue(bool(an.all()))
        expected = mass.sum() + .0012 * (p.y_xc * cons.sum() * p.dx * p.dy - p.delta * mass.sum())
        torch.testing.assert_close(mn.sum(), expected)
        mn.sum().backward()
        self.assertTrue(bool(torch.isfinite(mass.grad).all()))
        self.assertAlmostEqual(float(mass.grad[0]), 1 + .0012 * (p.mu0 / (p.ks + 1) - p.delta))

    def test_hotspot_records_and_shared_threshold_for_both_backends(self):
        p, n, dt = self.p, 12, .0012
        u, c = torch.full((n,), .5), torch.ones(n)
        noise = torch.zeros(n, 16, 2)
        for fn in (lambda **kw: simulate(u, c, p, n_steps=n, dt=dt, **kw),
                   lambda **kw: simulate_ibm(u, c, p, noise, n_steps=n, dt=dt, n_max=16, n0=8, **kw)):
            with self.subTest(backend=fn), torch.no_grad():
                out = fn()
                explicit = fn(t_ref=DEFAULT_T_REF)
                torch.testing.assert_close(out["temp_penalty"], explicit["temp_penalty"])
                torch.testing.assert_close(out["temp_penalty"], dt * (torch.tensor(out["records"]["temp_max"]) - DEFAULT_T_REF).clamp_min(0).square().sum())
                self.assertAlmostEqual(out["records"]["time"][-1], n * dt)
                self.assertEqual(out["peak_temp"], max(out["records"]["temp_max"]))
                self.assertGreater(out["peak_temp"], max(out["records"]["temp"]))
                sparse = fn(record_every=7)
                self.assertEqual(sparse["peak_temp"], out["peak_temp"])
                self.assertAlmostEqual(sparse["records"]["time"][-1], n * dt)

    def test_transport_stability_is_enforced(self):
        p = self.p
        with self.assertRaisesRegex(ValueError, "stability"):
            simulate(torch.ones(2), torch.ones(2), p, n_steps=2, dt=.01)
        self.assertLess(p.max_stable_dt(u_max=100), p.max_stable_dt())

    def test_autodiff_matches_finite_difference_for_both_backends(self):
        p, n, dt = self.p, 120, .0012
        noise = torch.randn(n, 32, 2, generator=torch.Generator().manual_seed(3))
        for ibm in (False, True):
            with self.subTest(ibm=ibm):
                ctrl = PiecewiseControl(3)
                fn = (lambda u, c: simulate_ibm(u, c, p, noise, n_steps=n, dt=dt, n_max=32, n0=8, record_every=n)) if ibm else (
                    lambda u, c: simulate(u, c, p, n_steps=n, dt=dt, record_every=n))
                # Check the physics path alone; a cost gradient could mask an error.
                def objective():
                    u, c = ctrl.step_values(n, dt, n * dt)
                    out = fn(u, c)
                    return out["x"].mean() + out["t"].square().mean()
                objective().backward()
                self.assertTrue(bool(torch.isfinite(ctrl.c_raw.grad).all()))
                ad, h = float(ctrl.c_raw.grad[0]), 1e-4
                with torch.no_grad():
                    original = float(ctrl.c_raw[0])
                    ctrl.c_raw[0] = original + h
                    plus = float(objective())
                    ctrl.c_raw[0] = original - h
                    minus = float(objective())
                    ctrl.c_raw[0] = original
                self.assertGreater(abs(ad), 1e-9)
                self.assertAlmostEqual(ad, (plus - minus) / (2 * h), delta=max(1e-10, abs(ad) * 1e-5))

    def test_time_step_refinement(self):
        means = []
        with torch.no_grad():
            for dt in (.0012, .0006):
                n = round(1.2 / dt)
                out = simulate(torch.full((n,), .5), torch.ones(n), self.p, n_steps=n, dt=dt, record_every=n)
                means.append(float(out["x"].mean()))
                self.assertTrue(bool(torch.isfinite(out["t"]).all()))
                self.assertGreaterEqual(float(out["c"].min()), 0.)
        self.assertLess(abs(means[0] - means[1]) / means[1], .001)


if __name__ == "__main__":
    unittest.main()
