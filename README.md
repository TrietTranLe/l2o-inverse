# L2O-Inverse: A Modular Learned Optimizer Framework for Inverse Problems
L2O-Inverse is a clean, modular, general-purpose Learned Optimizer (L2O) framework designed for solving inverse problems.  
While the provided example focuses on EEG Source Imaging (ESI), the framework is completely task-agnostic and can be used for any inverse problem.

Key properties:
- Modular components: grad_mod, update_rule, reg_net (via autoencoder)
- Loss defined symbolically via ExpressionLossComposer
- Fully configurable using Hydra
- Trains outer-loop with PyTorch Lightning
- Extensible and easy to maintain

---------------------------------------------------------------------

# 1. Quick Start

## Install dependencies
```
pip install -r requirements.txt
```

## Train a model
```
python -m scripts.train_from_config
```

Override configuration:
```
python -m scripts.train_from_config --config-name config_file_name
```

## Evaluate results
```
python -m scripts.eval_results_fsav -od /path/to/lightning_logs -mw checkpoint_file_name
```

## Visulize results
```
python -m scripts.visu_results_fsav -od /path/to/lightning_logs -mw checkpoint_file_name -i data_index
```

---------------------------------------------------------------------

# 2. Repository Structure

```
repo/
|
├── configs/
│   ├── baselines/              # baseline models for ESI
│   ├── dataset/                # dataset definitions
│   ├── loss/                   # inner_loss and outer_loss configs
│   ├── models/                 # autoencoder, grad_mod, update_rule, benchmark
│   ├── trainer/                # Lightning settings
│   └── config.yaml             # root config
|
├── data/
│   └──  eeg/                   # ESI data modules
|
├── exps/                       # experiment checkpoints and resultats
|
├── losses/
│   ├── builtins/               # MSE, L1, cosine, etc.
│   ├── expression_composer.py  # ExpressionLossComposer
│   └── README.md               # README.md for Loss System (ExpressionLossComposer)
|
├── models/
│   ├── benchmark/              # custom models for benchmark
│   ├── eeg/                    # baseline models for ESI
│   ├── autoencoder.py          # learned regularizer via autoencoder
│   ├── grad_mod.py             # gradient learning network
│   └── update_rule.py          # update rule
|
├── optimizers/
│   ├── benchmark/              # custom optimizer for benchmark
│   ├── builders/               # builders for the outer optimizer based on L2O submodules.
│   ├── l2o_optimizer.py        # inner optimization loop
│   └── outer_optimizer.py      # outer optimizer and learning rate schedulers
|
├── scripts/
│   ├── train_from_config.py    # train the model
│   ├── eval_results_fsav.py    # evaluate the results
│   └── visu_results_fsav.py    # visualize the results
|
├── tests/                      # test the module operation
|
├── trained_models/             # baseline checkpoints for ESI
|
├── trainers/
│   └── lit_bilevel.py          # outer optimization with PyTorch Lightning
|
├── scripts/
│   ├── train_from_config.py    # train the model
│   ├── eval_results_fsav.py    # evaluate the results
│   └── visu_results_fsav.py    # visualize the results
|
└── README.md
```

---------------------------------------------------------------------

# 3. Hydra Usage Guide

Override parameters:
```
python -m scripts.train_from_config l2o.n_steps=20 model.autoencoder.dim_hidden=512
```

Change config groups:
```
python -m scripts.train_from_config loss/inner_loss=inner_mse
```

Train benchmarks:
```
python -m scripts.train_from_config l2o.n_steps=20 model.autoencoder.dim_hidden=512
```

---------------------------------------------------------------------

# 4. Lightning Integration

The LitBiLevel module provides:
- inner-loop unrolling through L2O optimizer
- outer-loop meta-update
- per-group learning rates
- logging inner/outer losses
- checkpointing and device placement

---------------------------------------------------------------------

# 5. L2O Pipeline

## Inner Loop (optimize x)
```
x_0 = init
for k in 1..K:
    loss_inner = ExpressionLossComposer(x, y, L, AE)
    g = d(loss_inner)/dx
    g_mod = grad_mod(x, g)
    x = update_rule(x, g_mod)
```

## Outer Loop (optimize L2O parameters)
```
loss_outer = ExpressionLossComposer(x_K, x_true)
theta = outer_optimizer(loss_outer)
```

## Loss via symbolic expressions
Example YAML syntax:
```
expr: "fn(y, L @ x)"
expr: "fn(x, AE(x))"
```

---------------------------------------------------------------------

# 6. Example Loss Config

File: configs/loss/inner_loss/inner_cosine.yaml
```
_target_: losses.expression_composer.ExpressionLossComposer

terms:
  - name: data
    fn: { _target_: losses.builtins.cosine.CosineSimilarityFlatLoss }
    expr: "fn(y, L @ x)"
    weight: 1.0

  - name: reg
    fn: { _target_: losses.builtins.cosine.CosineSimilarityFlatLoss }
    expr: "fn(x, AE(x))"
    weight: 1.0

L:
  _target_: data.eeg.utils_eeg.build_leadfield_tensor
  forward_obj: ${fwd}
```

---------------------------------------------------------------------

# 7. Audience

This repository is intended for:
- researchers in inverse problems
- ML practitioners
- people exploring learned optimizers
- users wanting a plug-and-play configurable system
- non-experts who prefer YAML-level configuration without modifying code

---------------------------------------------------------------------

# 8. Citation

(Coming soon)

---------------------------------------------------------------------
