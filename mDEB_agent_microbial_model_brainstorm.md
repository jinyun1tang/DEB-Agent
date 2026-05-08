# Agent-Based Microbial Model Brainstorm Using mDEB

## Core Idea

Use the modified dynamic energy budget (mDEB) model as the within-cell physiology engine, and wrap it in an agent-based model (ABM) that represents microbial individuals, colonies, or super-individuals.

Each microbial agent can carry:

- Structural biomass `V`
- Reserve densities such as `xC`, `xN`, and optionally `xP` or other elements
- Trait parameters such as `vE`, uptake affinities, yields, maintenance costs, mortality thresholds, and division thresholds
- Optional position in space
- Optional strain or taxon identity
- Optional dormancy, enzyme allocation, motility, or stress state

At every timestep, each agent senses local substrates, updates reserves using mDEB, computes growth explicitly, changes structural biomass, and then may divide, die, respire, excrete, move, or enter dormancy.

## Physiology Loop

For each agent `a` and reserve `i`, use the mDEB structure:

```text
jA_i = uptake_i(S_i_local)
jE_i = vE_i * x_i

potential_growth_i = (jE_i - jM_i) / yV_i

mu_g = synthesizing_unit(potential_growth_C, potential_growth_N, ...)
mu_d = maintenance_deficit_loss
mu = mu_g - mu_d

dx_i/dt = jA_i - jE_i - mu * x_i + kappa_R * jR_i
dV/dt = mu * V
```

The key mDEB advantage is that reserve mobilization `jE_i` does not depend on `mu`, so growth can be computed directly without solving an implicit equation. That matters when the model has many agents.

## Agent Events

Natural ABM events include:

- Division when `V >= V_div`; split structure and reserves into two daughter agents, optionally with stochastic partitioning
- Death or lysis due to starvation, age, random mortality, stress, or `V < V_min`
- Release of dead biomass into dissolved or particulate organic matter pools
- Dormancy when maintenance deficits persist, with reduced uptake, turnover, and maintenance demand
- Phenotypic plasticity through adjustment of `vE`, uptake capacity, enzyme allocation, or reserve-use strategy
- Mutation or adaptation by perturbing inherited traits at division

## Environment Options

Start simple and add complexity in stages:

1. Chemostat ABM: well-mixed substrate pools with many agents competing for C and N.
2. Grid ABM: agents occupy cells in a 2D or 3D lattice; substrates diffuse and are consumed locally.
3. Particle or soil-aggregate model: agents live on organic matter particles and experience microsite heterogeneity.
4. Reactive transport coupling: substrate fields follow diffusion, advection, and reaction equations, with agents acting as biological source-sink terms.

## Why mDEB Fits an ABM

mDEB gives each agent internal memory through reserves. Agents do not respond only to instantaneous external substrate. They can keep growing after a substrate pulse ends, become stoichiometrically imbalanced, recover after nutrient changes, or diverge from one another because of their individual histories.

This is a strong match for ABMs, where heterogeneous histories and local interactions are central.

## Minimal Prototype

A good first version would be:

- Environment: well-mixed C and N pools
- Agent state: `V`, `xC`, `xN`, `alive`
- Physiology: adapt the official `mDEB.m` logic, but make parameters agent-specific or strain-specific
- Events: divide at `V >= 2 * V_birth`; die below reserve or structure thresholds
- Outputs: total biomass, POC, PON, substrate concentrations, reserve distributions, and trait distributions
- Validation: aggregate all agents and compare against the official chemostat mDEB trajectory

## Next Step

After the minimal chemostat version works, extend to a spatial ABM where local C and N pulses generate reserve-diverse microbial populations. The goal is to represent cells as chemically constrained organisms with internal state, rather than as memoryless Monod-growth particles.
