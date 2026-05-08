# Fitting an Individual-Based mDEB Model to Microfluidics Measurements

## Core Fit

Microfluidics is a natural match for an individual-based mDEB model because many measurements are already single-cell or lineage-resolved rather than bulk averages.

Represent each observed cell as an agent with hidden physiological state:

```text
agent state:
  V or cell length/area
  xC, xN reserve densities
  age or generation
  strain or phenotype
  local substrate exposure history
```

Microfluidics can provide time-lapse observations such as cell length, area, fluorescence, division timing, death, and local nutrient exposure. The individual-based mDEB model should predict those same observables through physiology plus division, death, dormancy, and inheritance rules.

## Measurements to Fit

The model could fit:

- Single-cell growth trajectories: length, area, or volume over time
- Interdivision times
- Birth size and division size
- Lineage-to-lineage variability
- Lag after nutrient shifts
- Growth recovery after starvation
- Mortality or dormancy probability
- Fluorescence reporters for C/N stress, ribosomes, enzymes, reserves, or metabolic state
- Spatial differences across channels, traps, or nutrient gradients

The most valuable information is not only the mean growth rate, but the full distribution of growth rates, division sizes, interdivision times, and recovery dynamics.

## Observation Model

The true cell state is hidden, so the model needs an observation layer:

```text
cell area or length ~ structural biomass V
fluorescence reporter ~ reserve density or stress state
division event occurs when V crosses a threshold
growth rate = d log(V) / dt
death or dormancy probability depends on reserve deficit
```

For rod-shaped bacteria, structural biomass can often be mapped to cell length if width is approximately constant. For yeast or irregular cells, segmented area or estimated volume may be a better proxy.

## Fitting Strategy

A practical workflow:

1. Segment cells and build lineage trees from microscopy.
2. Extract time series for `length(t)`, `area(t)`, fluorescence, division time, and death or disappearance events.
3. Run the individual-based mDEB model under the same nutrient schedule and flow conditions.
4. Fit parameters by matching both continuous trajectories and discrete event statistics.
5. Validate against held-out nutrient perturbations.

Useful parameters to estimate include:

- Reserve turnover `vE`
- Maximum uptake capacity `jm_A`
- Substrate affinity `K`
- Maintenance cost `jM`
- Biomass yield `yV`
- Rejected-reserve remobilization efficiency `kappa`
- Division threshold
- Stochastic partitioning noise at division
- Dormancy and starvation thresholds

## Why Individual Data Matter

Bulk growth curves can hide heterogeneity. Microfluidics can test whether the model captures:

- Asymmetric daughter-cell outcomes
- Persistence after starvation
- Delayed recovery after nutrient pulses
- Coexistence of fast and slow growers
- Reserve inheritance across generations
- Lineage memory after environmental change

These are exactly the behaviors where an individual-based mDEB model should be more informative than a bulk Monod-style growth model.

## Useful Experiments

Strong experiments for model calibration and validation:

- C pulse followed by N limitation
- N pulse after C-starved cells
- Alternating feast-famine cycles
- Constant substrate gradient across a microfluidic chamber
- Antibiotic or stress exposure followed by recovery
- Mother-machine lineage tracking over many generations

These experiments can help separate memory effects caused by internal reserves from effects caused by inherited traits, stochastic division, or local environmental heterogeneity.

## Model-Data Fit Target

Instead of fitting only mean growth, use a joint objective:

```text
loss =
  cell-size trajectory error
  + division-time distribution error
  + fluorescence trajectory error
  + lineage correlation error
  + population-size trajectory error
```

A strong validation test would be to train on one nutrient schedule and predict single-cell growth, division timing, and recovery distributions under a different pulse schedule. That would test whether the model is capturing physiology rather than only fitting curves.
