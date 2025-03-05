# MAPS $ ^2 $: Multi-Robot Autonomous Motion Planning under Signal Temporal Logic Specifications

[Preprint Link](https://arxiv.org/pdf/2309.05632)

The MAPS $ ^2 $ algorithm generates trajectories for multi-robot system under signal temporal logic constraints. The types of STL formulas currently supported are expressed as: $$ \varphi = \mu\ |\ \mathcal{G}_{[a,b]} \mu \ |\ \mathcal{F}_{[a,b]} \mu \ |\ \varphi_1 \land \varphi_2 \ |\ \mathcal{G}_{[a_1,b_1]}\mathcal{F}_{[a_2,b_2]} \mu \ |\ \mathcal{F}_{[a_1,b_1]}\mathcal{G}_{[a_2,b_2]} \mu $$

where $ \mu $ is any nonlinear predicate function predicate function.

## Prerequisites

Please ensure you have the following installed:
- Python 3.8 or later
- Required Python libraries: matplotlib, numpy, scipy (Install these via pip if not already installed)

```bash
pip install matplotlib numpy scipy
```

## Getting Started

Run the following script with an example STL formula in it. Configure the STL formula and the initial conditions as desired.

```
python run_script.py
```
