package numpy

// Cumsum returns a 1-D array of the running cumulative sum of the elements of a
// taken in logical row-major order. The result has the same number of elements
// as a.
func (a *NDArray) Cumsum() *NDArray {
	d := a.Data()
	out := make([]float64, len(d))
	var acc float64
	for i, v := range d {
		acc += v
		out[i] = acc
	}
	return newArray(out, []int{len(out)})
}

// Cumprod returns a 1-D array of the running cumulative product of the elements
// of a taken in logical row-major order. The result has the same number of
// elements as a.
func (a *NDArray) Cumprod() *NDArray {
	d := a.Data()
	out := make([]float64, len(d))
	acc := 1.0
	for i, v := range d {
		acc *= v
		out[i] = acc
	}
	return newArray(out, []int{len(out)})
}

// Diff returns the first discrete difference of a flattened 1-D copy of a, that
// is out[i] = a[i+1] - a[i]. The result has one fewer element than a; an array
// with 0 or 1 elements yields an empty result.
func (a *NDArray) Diff() *NDArray {
	d := a.Data()
	if len(d) <= 1 {
		return newArray([]float64{}, []int{0})
	}
	out := make([]float64, len(d)-1)
	for i := 0; i < len(out); i++ {
		out[i] = d[i+1] - d[i]
	}
	return newArray(out, []int{len(out)})
}

// Ptp returns the "peak to peak" range of a, that is Max minus Min. It panics
// on an empty array.
func (a *NDArray) Ptp() float64 {
	if a.size == 0 {
		panic("numpy: Ptp of empty array")
	}
	return a.Max() - a.Min()
}
