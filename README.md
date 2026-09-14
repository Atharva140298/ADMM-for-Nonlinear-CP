# ADMM for Nonlinear CP Decompositions (NCP)

Given a data tensor $\mathcal{X} \in \mathbb{R}^{I \times J \times K}$ and a
rank $R$, we solve the nonlinear CP decomposition problem

$$
\min_{ A \, B \, C \in \mathcal{C} \,\mathcal{T}}\ d\bigl(\mathcal{X} \, f(\mathcal{T})\bigr)
\quad \text{s.t.} \quad
\mathcal{T} = [\[A, B, C]\]
$$

where

* $f(\cdot)$ is an elementwise nonlinear function,
* $d(\cdot,\cdot)$ is the loss,
* $\mathcal{T} \in \mathbb{R}^{I \times J \times K}$ is a rank-$R$ tensor,
* $A \in \mathbb{R}^{I \times R}$, $B \in \mathbb{R}^{J \times R}$ and
  $C \in \mathbb{R}^{K \times R}$ are the factor matrices, and
  $[\[A,B,C]\] = \sum_{r=1}^{R} a_r \circ b_r \circ c_r$.

The factor matrices may additionally be constrained to be nonnegative.

## Nonlinear models $f(\mathcal{T})$

* **ReLU** &nbsp; $f(\mathcal{T}) = \max(0,\, \mathcal{T})$
* **CSF** (elementwise square) &nbsp; $f(\mathcal{T}) = \mathcal{T} \odot \mathcal{T}$
* **Min–Max** &nbsp; with bounds $a \le b$: &nbsp;
  $f(\mathcal{T}) = \min\bigl(b,\, \max(a,\, \mathcal{T})\bigr)$
* **Modulus** &nbsp; $f(\mathcal{T}) = |\mathcal{T}|$

## Loss functions $d(\mathcal{X}, f(\mathcal{T}))$

* **Frobenius norm**

$$\||\mathcal{X} - f(\mathcal{T})\||_F
= \sqrt{\sum_{i,j,k}\bigl(x_{ijk} - [f(\mathcal{T})]_{ijk}\bigr)^2}$$

* **$\ell_1$ norm**

$$\|\mathcal{X} - f(\mathcal{T})\|_1
= \sum_{i,j,k}\bigl|\,x_{ijk} - [f(\mathcal{T})]_{ijk}\,\bigr|$$

* **Kullback–Leibler (KL) divergence**, for nonnegative scalars $x, y \ge 0$:

$$
d_{\mathrm{KL}}(x,y) =
\begin{cases}
x\log\!\left(\dfrac{x}{y}\right) - x + y, & x > 0,\\[4pt]
y, & x = 0,
\end{cases}
\qquad
\mathrm{KL}(\mathcal{X}, f(\mathcal{T})) =
\sum_{i,j,k} d_{\mathrm{KL}}\bigl(x_{ijk},\, [f(\mathcal{T})]_{ijk}\bigr)
$$

## Algorithm

The problem is solved with ADMM. Introducing the auxiliary tensor
$\mathcal{T}$ decouples the nonlinearity from the low-rank structure, so that
$f$ and $d$ enter only through an entrywise subproblem, while the factor
updates reduce to (nonnegative) least-squares problems built from
Khatri–Rao products. Any combination of the nonlinearities and losses above
is therefore handled by the same algorithm.

## Requirements

* MATLAB
* [Tensor Toolbox](https://www.tensortoolbox.org/) — used for `cp_opt`,
  `cp_wopt`, `ktensor` and `score`
* [Tensorlab](https://www.tensorlab.net/) — used for `cpdgen`, `kr` and
  `tens2mat`

Both toolboxes must be on the MATLAB path before running any script.

## Usage

```matlab
param.maxiter = 1000;
param.tol     = 1e-5;
param.a       = 0;          % lower bound, for MinMax / ReLU
param.b       = 1;          % upper bound, for MinMax
param.factors = 'nonneg';   % or omit for unconstrained factors

res = admm_ntd(X, R, 'MinMax', 'Frobenius', param);
A = res.A;  B = res.B;  C = res.C;
```

## Contents

| file | description |
|---|---|
| `admm_ntd.m` | ADMM solver for the nonlinear CP model |
| `update_T.m` | entrywise subproblem for each nonlinearity and loss |
| `compute_objective_residual.m` | objective and residual evaluation |
| `cp_wopt_wridge.m` | CP-WOPT with ridge regularization (baseline) |
| `Synthetic_MinMax_sweep.m` | synthetic experiment: recovery vs clipping level |
| `NMR_sweep_censoring.m` | metabolomics experiment: recovery vs clipping level |
