package numpy

import "fmt"

// Flip returns a new contiguous array with the order of elements reversed along
// the given axis. Negative axis values count from the last axis. It panics if
// axis is out of range.
func (a *NDArray) Flip(axis int) *NDArray {
	if axis < 0 {
		axis += a.ndim
	}
	if axis < 0 || axis >= a.ndim {
		panic(fmt.Sprintf("numpy: Flip axis %d out of range for ndim %d", axis, a.ndim))
	}
	out := Zeros(a.shape...)
	n := a.shape[axis]
	a.forEach(func(off int, idx []int) {
		outIdx := make([]int, a.ndim)
		copy(outIdx, idx)
		outIdx[axis] = n - 1 - idx[axis]
		out.data[out.offset(outIdx)] = a.data[off]
	})
	return out
}

// Roll returns a new array the same shape as a with its flattened elements
// shifted circularly by shift positions (positive shifts move elements toward
// higher indices). The elements are taken in logical row-major order, rolled,
// and reshaped back, matching NumPy's roll with no axis argument.
func (a *NDArray) Roll(shift int) *NDArray {
	d := a.Data()
	n := len(d)
	out := make([]float64, n)
	if n == 0 {
		return newArray(out, a.shape)
	}
	s := ((shift % n) + n) % n
	for i := 0; i < n; i++ {
		out[(i+s)%n] = d[i]
	}
	return newArray(out, a.shape)
}

// Squeeze returns a view-independent copy of a with all axes of length 1
// removed. If every axis has length 1 the result is a 1-D array of length 1.
func (a *NDArray) Squeeze() *NDArray {
	newShape := make([]int, 0, a.ndim)
	for _, d := range a.shape {
		if d != 1 {
			newShape = append(newShape, d)
		}
	}
	if len(newShape) == 0 {
		newShape = []int{1}
	}
	return newArray(a.Data(), newShape)
}

// ExpandDims returns a copy of a with a new axis of length 1 inserted at the
// given position. axis may range from 0 to ndim inclusive; negative values
// count from the end (where -1 appends after the last existing axis).
func (a *NDArray) ExpandDims(axis int) *NDArray {
	if axis < 0 {
		axis += a.ndim + 1
	}
	if axis < 0 || axis > a.ndim {
		panic(fmt.Sprintf("numpy: ExpandDims axis %d out of range for ndim %d", axis, a.ndim))
	}
	newShape := make([]int, 0, a.ndim+1)
	newShape = append(newShape, a.shape[:axis]...)
	newShape = append(newShape, 1)
	newShape = append(newShape, a.shape[axis:]...)
	return newArray(a.Data(), newShape)
}
