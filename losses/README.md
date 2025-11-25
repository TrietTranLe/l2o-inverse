# Loss System (ExpressionLossComposer)

This folder contains the loss system used by the L2O/meta-learning framework.
It provides two main components:

1. **Built-in losses** (L1, L2, MSE, Cosine, etc.)
2. **ExpressionLossComposer** — a fully flexible, AST-based composite loss builder.

The composer allows defining loss functions directly from Hydra config using expressions such as:

```text
fn(y, L @ x)
fn(AE(x))
0.5 * fn(y, A @ x) + 0.1 * fn(x)
```

It is fully extensible and supports arbitrary variable names, arbitrary modules, and nested expressions.

---

## 1. ExpressionLossComposer Overview

`ExpressionLossComposer` is a PyTorch `nn.Module` that composes multiple loss terms defined in a config file.

Each term includes:

- **name**: identifier
- **fn**: a callable loss function module (e.g., MSE)
- **expr**: a Python-like expression such as `"fn(y, L @ x)"`  
- **weight** (optional):
  - if provided -> used as constant  
  - if omitted -> a trainable `nn.Parameter(1.0)` is created  

Variable names inside expressions have **no restrictions**.  
Examples: `AE`, `L`, `BigMatrix`, `PriorNetDeepV2`, `AAAAAE`, …

Expressions are parsed using the Python AST system and compiled once during initialization.

---

## 2. Hydra Integration (external_vars)

Variables (modules, tensors, etc.) can be injected directly from Hydra:

```yaml
loss:
  _target_: losses.expression_composer.ExpressionLossComposer
  AE: ${models.autoencoder}
  L: ${forward.leadfield}
  terms:
    - name: data
      fn: {_target_: losses.builtins.mse.MSE}
      expr: "fn(y, L @ x)"

    - name: reg
      fn: {_target_: losses.builtins.l1.L1}
      expr: "fn(x, AE(x))"
```

Variables declared at the top level (`AE:` , `L:`) become part of the composer context
and **do not need to be passed during training**.

Usage inside trainer:

```python
loss = composer(x=x_inner, y=target)
```

---

## 3. Expression Semantics

Supported syntax inside `expr`:

- function call: `fn(y, x)`
- module call: `AE(x)`
- batched matrix multiply: `A @ x`
- arithmetic: `+ - * /`
- nested expressions: `fn(y, A @ x) + 0.1 * fn(x)`
- any variable name allowed

Expressions are parsed with AST and compiled once, so evaluating loss is very fast.

---

## 4. Getting Loss Values

Compute the total loss:

```python
loss = composer(x=x, y=y)
```

Compute total + weighted terms:

```python
loss, wterms = composer(x=x, y=y, return_terms=True)
```

Compute raw (unweighted) terms:

```python
composer.compute_raw_terms(x=x, y=y)
```

---

## 5. Built-in Loss Functions

Located in:

```
losses/builtins/
```

Each file contains a small PyTorch `nn.Module`. Examples:

- `MSE`
- `L1`
- `CosineSimilarity`
- `CosineReshape`
- etc.

You can add your own by simply creating new modules.

---

## 6. Defining a Custom Loss Term

A term is defined as:

```yaml
- name: prior
  fn: {_target_: losses.builtins.l1.L1}
  expr: "fn(x)"
  weight: 0.5
```

If `weight` is omitted:

```
nn.Parameter(1.0)  # trainable weight created automatically
```

---

## 7. Summary

`ExpressionLossComposerV4.1` provides:

- AST-based composite loss expressions
- Hydra integration via `external_vars`
- trainable or fixed term weights
- arbitrary variable names
- no registry required
- compiled once (fast inner-loop execution)
- raw + weighted term extraction for logging

It enables flexible, modular loss definitions entirely from config,  
without writing new Python code for each experiment.
