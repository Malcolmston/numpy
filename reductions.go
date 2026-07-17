package numpy

import (
	"fmt"
	"math"
)

// Sum returns the sum of all elements.
func (a *NDArray) Sum() float64 {
	var s float64
	a.forEach(func(off int, _ []int) { s += a.data[off] })
	return s
}

// Prod returns the product of all elements.
func (a *NDArray) Prod() float64 {
	p := 1.0
	a.forEach(func(off int, _ []int) { p *= a.data[off] })
	return p
}

// Mean returns the arithmetic mean of all elements. It panics on an empty array.
func (a *NDArray) Mean() float64 {
	if a.size == 0 {
		panic("numpy: Mean of empty array")
	}
	return a.Sum() / float64(a.size)
}

// Max returns the largest element. It panics on an empty array.
func (a *NDArray) Max() float64 {
	if a.size == 0 {
		panic("numpy: Max of empty array")
	}
	m := math.Inf(-1)
	a.forEach(func(off int, _ []int) {
		if a.data[off] > m {
			m = a.data[off]
		}
	})
	return m
}

// Min returns the smallest element. It panics on an empty array.
func (a *NDArray) Min() float64 {
	if a.size == 0 {
		panic("numpy: Min of empty array")
	}
	m := math.Inf(1)
	a.forEach(func(off int, _ []int) {
		if a.data[off] < m {
			m = a.data[off]
		}
	})
	return m
}

// Var returns the population variance of all elements (dividing by N).
func (a *NDArray) Var() float64 {
	mean := a.Mean()
	var s float64
	a.forEach(func(off int, _ []int) {
		d := a.data[off] - mean
		s += d * d
	})
	return s / float64(a.size)
}

// Std returns the population standard deviation of all elements.
func (a *NDArray) Std() float64 { return math.Sqrt(a.Var()) }

// reduceAxis reduces along one axis using an accumulator produced by init,
// combined with combine, and finalized with finish. When keepdims is true the
// reduced axis is retained with length 1.
func (a *NDArray) reduceAxis(axis int, keepdims bool, init func() float64, combine func(acc, x float64) float64, finish func(acc float64, n int) float64) *NDArray {
	if axis < 0 {
		axis += a.ndim
	}
	if axis < 0 || axis >= a.ndim {
		panic(fmt.Sprintf("numpy: reduction axis %d out of range for ndim %d", axis, a.ndim))
	}
	// Result shape drops (or keeps as 1) the reduced axis.
	outShape := make([]int, 0, a.ndim)
	for d := 0; d < a.ndim; d++ {
		if d == axis {
			if keepdims {
				outShape = append(outShape, 1)
			}
			continue
		}
		outShape = append(outShape, a.shape[d])
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	out := Zeros(outShape...)
	acc := make([]float64, out.size)
	for i := range acc {
		acc[i] = init()
	}
	n := a.shape[axis]
	// Map each source element to its output position by removing the reduced axis.
	a.forEach(func(off int, idx []int) {
		outIdx := make([]int, 0, len(outShape))
		for d := 0; d < a.ndim; d++ {
			if d == axis {
				if keepdims {
					outIdx = append(outIdx, 0)
				}
				continue
			}
			outIdx = append(outIdx, idx[d])
		}
		if len(outIdx) == 0 {
			outIdx = []int{0}
		}
		pos := out.offset(outIdx)
		acc[pos] = combine(acc[pos], a.data[off])
	})
	for i := range acc {
		out.data[i] = finish(acc[i], n)
	}
	return out
}

// SumAxis reduces a by summation along the given axis.
func (a *NDArray) SumAxis(axis int, keepdims bool) *NDArray {
	return a.reduceAxis(axis, keepdims,
		func() float64 { return 0 },
		func(acc, x float64) float64 { return acc + x },
		func(acc float64, _ int) float64 { return acc })
}

// ProdAxis reduces a by product along the given axis.
func (a *NDArray) ProdAxis(axis int, keepdims bool) *NDArray {
	return a.reduceAxis(axis, keepdims,
		func() float64 { return 1 },
		func(acc, x float64) float64 { return acc * x },
		func(acc float64, _ int) float64 { return acc })
}

// MeanAxis reduces a by mean along the given axis.
func (a *NDArray) MeanAxis(axis int, keepdims bool) *NDArray {
	return a.reduceAxis(axis, keepdims,
		func() float64 { return 0 },
		func(acc, x float64) float64 { return acc + x },
		func(acc float64, n int) float64 { return acc / float64(n) })
}

// MaxAxis reduces a by maximum along the given axis.
func (a *NDArray) MaxAxis(axis int, keepdims bool) *NDArray {
	return a.reduceAxis(axis, keepdims,
		func() float64 { return math.Inf(-1) },
		math.Max,
		func(acc float64, _ int) float64 { return acc })
}

// MinAxis reduces a by minimum along the given axis.
func (a *NDArray) MinAxis(axis int, keepdims bool) *NDArray {
	return a.reduceAxis(axis, keepdims,
		func() float64 { return math.Inf(1) },
		math.Min,
		func(acc float64, _ int) float64 { return acc })
}

// StdAxis reduces a by population standard deviation along the given axis.
func (a *NDArray) StdAxis(axis int, keepdims bool) *NDArray {
	mean := a.MeanAxis(axis, true)
	// Broadcast the mean back over a, square the deviations, then mean+sqrt.
	dev := a.Sub(mean)
	sq := dev.Mul(dev)
	variance := sq.MeanAxis(axis, keepdims)
	return variance.Sqrt()
}
