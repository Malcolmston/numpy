package numpy

import (
	"fmt"
	"strings"
)

// NDArray is an n-dimensional, row-major array of float64 values.
//
// The element data is held in a single flat slice. The logical view of that
// data is described by shape (the length along each axis) and strides (the
// number of flat elements to step to advance one position along each axis).
// A freshly created array is always contiguous in row-major (C) order, but
// views produced by Transpose or slicing may be non-contiguous.
//
// All shape-validation failures panic with a clear message. This keeps the
// arithmetic API free of error returns and lets operations be chained. Use
// recover if you need to trap these panics.
type NDArray struct {
	data    []float64
	shape   []int
	strides []int
	ndim    int
	size    int
}

// rowMajorStrides returns the row-major (C-order) strides for a shape.
func rowMajorStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// sizeOf returns the product of the dimensions of shape (1 for a scalar).
func sizeOf(shape []int) int {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return size
}

// newArray builds an NDArray owning data with the given shape. It computes
// row-major strides and validates that len(data) matches the shape size.
func newArray(data []float64, shape []int) *NDArray {
	for _, d := range shape {
		if d < 0 {
			panic(fmt.Sprintf("numpy: negative dimension in shape %v", shape))
		}
	}
	size := sizeOf(shape)
	if len(data) != size {
		panic(fmt.Sprintf("numpy: data length %d does not match shape %v (size %d)", len(data), shape, size))
	}
	sc := make([]int, len(shape))
	copy(sc, shape)
	return &NDArray{
		data:    data,
		shape:   sc,
		strides: rowMajorStrides(sc),
		ndim:    len(sc),
		size:    size,
	}
}

// Ndim returns the number of dimensions (axes) of the array.
func (a *NDArray) Ndim() int { return a.ndim }

// Size returns the total number of elements in the array.
func (a *NDArray) Size() int { return a.size }

// Shape returns a copy of the array's shape.
func (a *NDArray) Shape() []int {
	s := make([]int, len(a.shape))
	copy(s, a.shape)
	return s
}

// Strides returns a copy of the array's strides (in elements, not bytes).
func (a *NDArray) Strides() []int {
	s := make([]int, len(a.strides))
	copy(s, a.strides)
	return s
}

// isContiguous reports whether the array is laid out in row-major order with
// no gaps, so that its logical order equals a.data read front to back.
func (a *NDArray) isContiguous() bool {
	expected := rowMajorStrides(a.shape)
	for i := range expected {
		// Dimensions of length 1 may carry any stride; ignore them.
		if a.shape[i] == 1 {
			continue
		}
		if a.strides[i] != expected[i] {
			return false
		}
	}
	return true
}

// offset converts a multi-index into a flat offset into a.data.
func (a *NDArray) offset(idx []int) int {
	if len(idx) != a.ndim {
		panic(fmt.Sprintf("numpy: index %v has wrong number of dimensions for shape %v", idx, a.shape))
	}
	off := 0
	for axis, i := range idx {
		if i < 0 {
			i += a.shape[axis]
		}
		if i < 0 || i >= a.shape[axis] {
			panic(fmt.Sprintf("numpy: index %d out of range for axis %d with size %d", idx[axis], axis, a.shape[axis]))
		}
		off += i * a.strides[axis]
	}
	return off
}

// At returns the element at the given multi-index. Negative indices count
// from the end of each axis.
func (a *NDArray) At(idx ...int) float64 {
	return a.data[a.offset(idx)]
}

// Set stores v at the given multi-index. Negative indices count from the end.
func (a *NDArray) Set(v float64, idx ...int) {
	a.data[a.offset(idx)] = v
}

// Data returns a copy of the array's elements in logical row-major order.
// The result is always contiguous regardless of the array's internal layout.
func (a *NDArray) Data() []float64 {
	out := make([]float64, a.size)
	i := 0
	a.forEach(func(off int, _ []int) {
		out[i] = a.data[off]
		i++
	})
	return out
}

// forEach visits every element in logical row-major order, calling fn with the
// flat offset into a.data and the current multi-index. The index slice is
// reused between calls and must not be retained.
func (a *NDArray) forEach(fn func(off int, idx []int)) {
	if a.size == 0 {
		return
	}
	idx := make([]int, a.ndim)
	for {
		off := 0
		for axis, i := range idx {
			off += i * a.strides[axis]
		}
		fn(off, idx)
		// Increment the multi-index (row-major, last axis fastest).
		axis := a.ndim - 1
		for axis >= 0 {
			idx[axis]++
			if idx[axis] < a.shape[axis] {
				break
			}
			idx[axis] = 0
			axis--
		}
		if axis < 0 {
			return
		}
	}
}

// Copy returns a new contiguous array holding a copy of the data.
func (a *NDArray) Copy() *NDArray {
	return newArray(a.Data(), a.shape)
}

// Equal reports whether two arrays have the same shape and identical elements.
func (a *NDArray) Equal(b *NDArray) bool {
	if a.ndim != b.ndim {
		return false
	}
	for i := range a.shape {
		if a.shape[i] != b.shape[i] {
			return false
		}
	}
	ad, bd := a.Data(), b.Data()
	for i := range ad {
		if ad[i] != bd[i] {
			return false
		}
	}
	return true
}

// AllClose reports whether two arrays have the same shape and all elements
// differ by no more than tol in absolute value.
func (a *NDArray) AllClose(b *NDArray, tol float64) bool {
	if a.ndim != b.ndim {
		return false
	}
	for i := range a.shape {
		if a.shape[i] != b.shape[i] {
			return false
		}
	}
	ad, bd := a.Data(), b.Data()
	for i := range ad {
		d := ad[i] - bd[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			return false
		}
	}
	return true
}

// String renders the array's shape and data for debugging.
func (a *NDArray) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "NDArray(shape=%v, data=%v)", a.shape, a.Data())
	return b.String()
}
