# DEB-Agent

Research workspace for exploring modified Dynamic Energy Budget (mDEB) models of microbial metabolism, with an emphasis on agent-based microbial physiology and a Crabtree-effect model sketch.

The repository currently contains model notes, reference papers, an executable Jupyter notebook, an HTML model sketch, and presentation artifacts. It is best treated as a living modeling sandbox rather than a packaged software library.

## Current Focus

The most developed artifact is a Crabtree-effect model that couples:

- mDEB reserve dynamics for substrate uptake, reserve mobilization, maintenance, and living structural biomass.
- Ohm's-law-inspired pathway returns for fermentation and respiration.
- A dynamic softmax splitter that allocates mobilized carbon between fermentation and respiration.
- Ethanol as both an overflow product and a later respiratory substrate once glucose repression relaxes.
- Mortality or negative structural growth when catabolic supply is below maintenance demand.

The model is intended to test qualitative batch-culture behavior: glucose drawdown, ethanol overflow, oxygen drawdown and recovery, diauxic ethanol use, total-biomass plateau, and living structural biomass decline under maintenance stress.

## Repository Layout

```text
.
|-- references/
|   |-- crabtree-effect-model-sketch.html
|   |-- mDEB_agent_microbial_model_brainstorm.md
|   |-- mDEB_microfluidics_individual_model_brainstorm.md
|   `-- *.pdf
|-- src/
|   `-- crabtree-effect-mdeb-model.ipynb
|-- outputs/
|   |-- crabtree-model-presentation.pptx
|   |-- crabtree-model-presentation/
|   `-- crabtree-model-deck-build/
`-- presentation/
    `-- crabtree-model-presentation.pptx
```

## Key Artifacts

### Crabtree HTML model sketch

Open:

```text
references/crabtree-effect-model-sketch.html
```

This file gives the conceptual model in a browser-friendly form. It includes the state diagram, equations, symbol table, softmax splitter rationale, attention-mechanism analogy, and the update that treats glucose and ethanol as distinct substrates.

### Crabtree Jupyter notebook

Open:

```text
src/crabtree-effect-mdeb-model.ipynb
```

This notebook implements an illustrative Python version of the Crabtree model. The default parameter set is for structure testing, not a fitted yeast dataset. It simulates a high-glucose batch or pulse scenario and visualizes external substrate, oxygen, reserve, living structural biomass, total biomass proxy, fermentation allocation, ethanol, and mortality diagnostics.

Minimal Python dependencies:

```text
numpy
IPython  # optional, for SVG display in notebooks
```

### Crabtree presentation

Current presentation output:

```text
outputs/crabtree-model-presentation.pptx
presentation/crabtree-model-presentation.pptx
```

The deck introduces the Crabtree effect, states the modeling hypothesis, defines evaluation metrics, summarizes equations, and shows the softmax pathway-transition graph and notebook results.

The reproducible deck build workspace is:

```text
outputs/crabtree-model-deck-build/
```

The main generator script is:

```text
outputs/crabtree-model-deck-build/build-crabtree-deck.mjs
```

## Model Sketch

The reduced Crabtree model tracks:

- `G`: external glucose.
- `EtOH`: external ethanol, produced by fermentation and later consumed as substrate.
- `O2`: dissolved oxygen.
- `x_C`: internal carbon reserve density.
- `V`: living structural biomass.
- `theta_F`: fermentation allocation fraction.

Glucose enters reserve through a Monod-like uptake term. Reserve is mobilized and split between fermentation and respiration. The target allocation is a two-pathway softmax:

```text
theta_i* = exp(U_i) / [exp(U_F) + exp(U_R)]
```

For the two-pathway case:

```text
theta_F* = 1 / [1 + exp(-(U_F - U_R))]
theta_R* = 1 - theta_F*
```

The actual allocation relaxes toward the target:

```text
dtheta_F/dt = (theta_F* - theta_F) / tau_reg
```

This dynamic splitter is useful because it represents a smooth, lagged regulatory transition rather than an instantaneous threshold switch.

## Evaluation Targets

Useful batch-culture metrics include:

- Glucose depletion time and uptake rate.
- Ethanol peak, production slope, and consumption slope.
- Dissolved oxygen minimum and recovery time.
- Living structural biomass `V` versus total biomass proxy `V(1 + x_C)`.
- Timing and hysteresis of `theta_F`.
- Mortality or negative growth when maintenance demand exceeds catabolic supply.

These metrics are meant to keep the model falsifiable: the goal is not only to generate ethanol, but to reproduce the full observed sequence of overflow, diauxic transition, and stationary or death phase behavior.

## Reference Notes

The Crabtree model draws most directly from:

- Tang and Riley (2025), `references/bg-22-1809-2025.pdf`, for the modified Dynamic Energy Budget formulation.
- Tang et al. (2021), `references/2021-8 copy.pdf`, for the Ohm's-law analogy and discussion of Warburg/Crabtree-like metabolism.

Additional brainstorming notes:

- `references/mDEB_agent_microbial_model_brainstorm.md`: agent-based microbial model using mDEB as the within-cell physiology engine.
- `references/mDEB_microfluidics_individual_model_brainstorm.md`: fitting individual-based mDEB models to microfluidics and lineage-resolved observations.

## Suggested Next Steps

1. Update the notebook so ethanol uptake is explicitly oxygen dependent and glucose repressed, matching the HTML sketch.
2. Add an oxygen-demand cap or flux limiter so dissolved oxygen cannot become negative by construction.
3. Add parameter sweeps for `tau_reg`, `beta_a`, oxygen transfer, and glucose pulse size.
4. Compare model outputs against real batch-culture data using the evaluation metrics above.
5. Separate generated build intermediates from source artifacts if this repository becomes a cleaner package.

## Status

This project is exploratory. Equations and parameters are currently designed for conceptual testing and model development. Treat default outputs as diagnostics of model structure, not calibrated predictions.
