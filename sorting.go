package numpy

import "sort"

// Sort returns a new 1-D array containing all elements of a in ascending order.
// The array is flattened in logical row-major order before sorting, mirroring
// NumPy's sort of a flattened array. NaN values sort to the end.
func (a *NDArray) Sort() *NDArray {
	d := a.Data()
	sort.Float64s(d)
	return newArray(d, []int{len(d)})
}

// Argsort returns the indices that would sort a flattened copy of the array in
// ascending order, as a 1-D float64 array. The sort is stable, so equal
// elements keep their original relative order.
func (a *NDArray) Argsort() *NDArray {
	d := a.Data()
	idx := make([]int, len(d))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return d[idx[i]] < d[idx[j]] })
	out := make([]float64, len(idx))
	for i, v := range idx {
		out[i] = float64(v)
	}
	return newArray(out, []int{len(out)})
}

// Argmax returns the flat index (in logical row-major order) of the first
// maximum element. It panics on an empty array.
func (a *NDArray) Argmax() int {
	if a.size == 0 {
		panic("numpy: Argmax of empty array")
	}
	d := a.Data()
	best := 0
	for i := 1; i < len(d); i++ {
		if d[i] > d[best] {
			best = i
		}
	}
	return best
}

// Argmin returns the flat index (in logical row-major order) of the first
// minimum element. It panics on an empty array.
func (a *NDArray) Argmin() int {
	if a.size == 0 {
		panic("numpy: Argmin of empty array")
	}
	d := a.Data()
	best := 0
	for i := 1; i < len(d); i++ {
		if d[i] < d[best] {
			best = i
		}
	}
	return best
}

// Unique returns a new 1-D array of the distinct element values of a, sorted in
// ascending order.
func (a *NDArray) Unique() *NDArray {
	d := a.Data()
	sort.Float64s(d)
	out := make([]float64, 0, len(d))
	for i, v := range d {
		if i == 0 || v != d[i-1] {
			out = append(out, v)
		}
	}
	return newArray(out, []int{len(out)})
}

// SearchSorted returns the leftmost index where v could be inserted into a to
// keep it sorted in ascending order. The receiver must be a 1-D array that is
// already sorted; the result lies in the range [0, len]. It panics if a is not
// 1-D.
func (a *NDArray) SearchSorted(v float64) int {
	if a.ndim != 1 {
		panic("numpy: SearchSorted requires a 1-D array")
	}
	return sort.SearchFloat64s(a.Data(), v)
}
