# Changelog

All notable changes to this project are documented in this file.

## [0.2.0]

Added a large batch of NumPy-parity functionality (all standard-library only,
deterministic, and covered by known-answer tests).

### Element-wise math (`mathextra.go`)
- New unary methods: `Tan`, `Sinh`, `Cosh`, `Tanh`, `Arcsin`, `Arccos`,
  `Arctan`, `Floor`, `Ceil`, `Round` (round-half-to-even), `Trunc`, `Sign`,
  `Square`, `Reciprocal`, `Cbrt`, `Log2`, `Log10`, `Log1p`, `Expm1`.
- New binary (broadcasting) methods: `Arctan2`, `Hypot`, `Mod`.
- New `Clip(min, max)` method.

### Sorting and searching (`sorting.go`)
- `Sort`, `Argsort`, `Argmax`, `Argmin`, `Unique`, `SearchSorted`.

### Cumulative operations (`accumulate.go`)
- `Cumsum`, `Cumprod`, `Diff`, `Ptp`.

### Statistics (`statistics.go`)
- `Median`, `Percentile`, `Quantile`, `VarDDof`, `StdDDof` (delta degrees of
  freedom for sample vs. population variance/standard deviation).

### Linear algebra (`linalgextra.go`)
- `Trace`, `Diagonal`, `Outer`, `Cross`, `Norm` (Frobenius/L2), and the
  package-level `Diag` (build a diagonal matrix from a vector, or extract the
  diagonal of a matrix).

### Array manipulation (`manipextra.go`)
- `Flip`, `Roll`, `Squeeze`, `ExpandDims`.

## [0.1.0]

Initial release: the dense `NDArray` type with creation helpers, shape
manipulation, broadcasting element-wise math, axis reductions, basic linear
algebra (`Dot`, `MatMul`), and boolean masking.
