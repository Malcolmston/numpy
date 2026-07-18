package numpy

import (
	"math"
	"sort"
)

// Median returns the median of all elements. For an even number of elements it
// returns the average of the two middle values. It panics on an empty array.
func (a *NDArray) Median() float64 {
	if a.size == 0 {
		panic("numpy: Median of empty array")
	}
	d := a.Data()
	sort.Float64s(d)
	n := len(d)
	if n%2 == 1 {
		return d[n/2]
	}
	return (d[n/2-1] + d[n/2]) / 2
}

// Percentile returns the q-th percentile of all elements, with q in the range
// [0, 100]. It uses linear interpolation between the two nearest ranks, the
// same as NumPy's default ("linear") method. It panics on an empty array or if
// q is outside [0, 100].
func (a *NDArray) Percentile(q float64) float64 {
	if a.size == 0 {
		panic("numpy: Percentile of empty array")
	}
	if q < 0 || q > 100 {
		panic("numpy: Percentile q must be in [0, 100]")
	}
	d := a.Data()
	sort.Float64s(d)
	n := len(d)
	if n == 1 {
		return d[0]
	}
	rank := q / 100 * float64(n-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return d[lo]
	}
	frac := rank - float64(lo)
	return d[lo]*(1-frac) + d[hi]*frac
}

// Quantile returns the q-th quantile of all elements, with q in the range
// [0, 1]. It is equivalent to Percentile(q*100).
func (a *NDArray) Quantile(q float64) float64 {
	return a.Percentile(q * 100)
}

// VarDDof returns the variance of all elements using the given delta degrees of
// freedom: the sum of squared deviations from the mean is divided by
// (N - ddof). ddof=0 gives the population variance (like Var) and ddof=1 gives
// the unbiased sample variance. It panics if N-ddof <= 0.
func (a *NDArray) VarDDof(ddof int) float64 {
	denom := a.size - ddof
	if denom <= 0 {
		panic("numpy: VarDDof requires N-ddof > 0")
	}
	mean := a.Mean()
	var s float64
	a.forEach(func(off int, _ []int) {
		delta := a.data[off] - mean
		s += delta * delta
	})
	return s / float64(denom)
}

// StdDDof returns the standard deviation of all elements using the given delta
// degrees of freedom; it is the square root of VarDDof(ddof).
func (a *NDArray) StdDDof(ddof int) float64 {
	return math.Sqrt(a.VarDDof(ddof))
}
