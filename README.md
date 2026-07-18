# numpy

N-dimensional numeric arrays for Go — a small, standard-library-only library
modeled on the core of Python's NumPy.

It provides a single dense `NDArray` of `float64`, backed by a flat slice with
row-major shape and strides, plus creation helpers, broadcasting element-wise
math, axis reductions, basic linear algebra, and boolean masking. No cgo, no
third-party dependencies.

## Install

```sh
go get github.com/malcolmston/numpy
```

Requires Go 1.24 or newer.

## Quick start

```go
package main

import (
	"fmt"

	np "github.com/malcolmston/numpy"
)

func main() {
	// Create and reshape.
	a := np.Arange(0, 6, 1).Reshape(2, 3) // [[0 1 2] [3 4 5]]

	// Broadcasting: (2,3) + (3,) -> (2,3).
	b := np.FromSlice([]float64{10, 20, 30})
	fmt.Println(a.Add(b).Data()) // [10 21 32 13 24 35]

	// Reductions along an axis.
	fmt.Println(a.SumAxis(0, false).Data()) // [3 5 7]
	fmt.Println(a.Mean())                   // 2.5

	// Matrix multiplication.
	m := np.FromData([]float64{1, 2, 3, 4}, 2, 2)
	fmt.Println(m.MatMul(m).Data()) // [7 10 15 22]

	// Boolean masking.
	mask := a.GreaterScalar(2)
	fmt.Println(a.MaskSelect(mask).Data()) // [3 4 5]
}
```

## Overview

- **Creation:** `FromSlice`, `FromData`, `FromNested`, `Zeros`, `Ones`, `Full`,
  `Arange`, `Linspace`, `Eye`, `Identity`, `ZerosLike`, `OnesLike`.
- **Manipulation:** `Reshape`, `Ravel`/`Flatten`, `Transpose`/`T`, `Slice`,
  `Concatenate`, `Stack`, `At`, `Set`.
- **Element-wise math:** `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`, `Abs`,
  `Sqrt`, `Exp`, `Log`, `Sin`, `Cos`, and `*Scalar` variants, plus `Maximum`
  and `Minimum`.
- **Reductions:** `Sum`, `Mean`, `Max`, `Min`, `Std`, `Var`, `Prod` over the
  whole array, and `SumAxis`, `MeanAxis`, `MaxAxis`, `MinAxis`, `StdAxis`,
  `ProdAxis` along an axis (with optional `keepdims`).
- **More element-wise math:** `Tan`, `Sinh`, `Cosh`, `Tanh`, `Arcsin`,
  `Arccos`, `Arctan`, `Arctan2`, `Hypot`, `Mod`, `Floor`, `Ceil`, `Round`,
  `Trunc`, `Sign`, `Square`, `Reciprocal`, `Cbrt`, `Log2`, `Log10`, `Log1p`,
  `Expm1`, and `Clip`.
- **Sorting and searching:** `Sort`, `Argsort`, `Argmax`, `Argmin`, `Unique`,
  `SearchSorted`.
- **Cumulative ops:** `Cumsum`, `Cumprod`, `Diff`, `Ptp`.
- **Statistics:** `Median`, `Percentile`, `Quantile`, `VarDDof`, `StdDDof`.
- **Linear algebra:** `Dot` (1-D dot / 2-D matmul), `MatMul`, `Trace`,
  `Diagonal`, `Outer`, `Cross`, `Norm`, and `Diag`.
- **More manipulation:** `Flip`, `Roll`, `Squeeze`, `ExpandDims`.
- **Comparison and masking:** `Greater`, `GreaterEqual`, `Less`, `LessEqual`,
  `EqualMask`, `NotEqualMask` (and scalar variants), `MaskSelect`, `Where`,
  `Any`, `All`.

Broadcasting follows NumPy's rules: shapes are aligned from the trailing axis
and dimensions must be equal or one of them must be 1.

## Errors

All shape and dimension validation failures panic with a message prefixed by
`numpy:`. This keeps the arithmetic API free of error returns so operations can
be chained. Wrap calls with `recover` if you need to trap invalid usage.

## Version

See the `VERSION` file. Current version: `0.2.0`. See `CHANGELOG.md` for the
list of changes.
