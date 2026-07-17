package numpy

import "fmt"

// Reshape returns a new array with the same data viewed under a new shape.
// Exactly one dimension may be -1, in which case it is inferred from the total
// size. The data is copied to contiguous order first, so the result is always
// independent of the receiver.
func (a *NDArray) Reshape(shape ...int) *NDArray {
	resolved := make([]int, len(shape))
	copy(resolved, shape)
	inferAxis := -1
	known := 1
	for i, d := range resolved {
		if d == -1 {
			if inferAxis != -1 {
				panic("numpy: Reshape allows only one inferred (-1) dimension")
			}
			inferAxis = i
			continue
		}
		if d < 0 {
			panic(fmt.Sprintf("numpy: invalid negative dimension %d in Reshape", d))
		}
		known *= d
	}
	if inferAxis != -1 {
		if known == 0 || a.size%known != 0 {
			panic(fmt.Sprintf("numpy: cannot infer -1 dimension reshaping size %d to %v", a.size, shape))
		}
		resolved[inferAxis] = a.size / known
	}
	if sizeOf(resolved) != a.size {
		panic(fmt.Sprintf("numpy: cannot reshape array of size %d into shape %v", a.size, shape))
	}
	return newArray(a.Data(), resolved)
}

// Ravel returns a contiguous 1-D copy of the array in row-major order.
func (a *NDArray) Ravel() *NDArray {
	return newArray(a.Data(), []int{a.size})
}

// Flatten is an alias for Ravel; it returns a contiguous 1-D copy.
func (a *NDArray) Flatten() *NDArray { return a.Ravel() }

// Transpose returns a view with the axes permuted. With no arguments the axis
// order is reversed. Otherwise axes must be a permutation of 0..ndim-1. The
// result shares data with the receiver (it is a view, not a copy).
func (a *NDArray) Transpose(axes ...int) *NDArray {
	if len(axes) == 0 {
		axes = make([]int, a.ndim)
		for i := range axes {
			axes[i] = a.ndim - 1 - i
		}
	}
	if len(axes) != a.ndim {
		panic(fmt.Sprintf("numpy: Transpose axes %v do not match ndim %d", axes, a.ndim))
	}
	seen := make([]bool, a.ndim)
	newShape := make([]int, a.ndim)
	newStrides := make([]int, a.ndim)
	for i, ax := range axes {
		if ax < 0 || ax >= a.ndim || seen[ax] {
			panic(fmt.Sprintf("numpy: invalid Transpose axes %v", axes))
		}
		seen[ax] = true
		newShape[i] = a.shape[ax]
		newStrides[i] = a.strides[ax]
	}
	return &NDArray{
		data:    a.data,
		shape:   newShape,
		strides: newStrides,
		ndim:    a.ndim,
		size:    a.size,
	}
}

// T is shorthand for a full transpose (reverse of all axes).
func (a *NDArray) T() *NDArray { return a.Transpose() }

// Slice returns a view selecting a contiguous range [start, stop) along each
// axis. Provide one Range per axis. A zero Range value (Start and Stop both 0)
// is treated as the full extent of that axis. Negative Start/Stop count from
// the end. The result shares data with the receiver.
func (a *NDArray) Slice(ranges ...Range) *NDArray {
	if len(ranges) != a.ndim {
		panic(fmt.Sprintf("numpy: Slice needs %d ranges, got %d", a.ndim, len(ranges)))
	}
	newShape := make([]int, a.ndim)
	newStrides := make([]int, a.ndim)
	offset := 0
	for axis, r := range ranges {
		start, stop := r.resolve(a.shape[axis])
		newShape[axis] = stop - start
		newStrides[axis] = a.strides[axis]
		offset += start * a.strides[axis]
	}
	return &NDArray{
		data:    a.data[offset:],
		shape:   newShape,
		strides: newStrides,
		ndim:    a.ndim,
		size:    sizeOf(newShape),
	}
}

// Range describes a half-open [Start, Stop) selection along one axis for Slice.
// If both fields are zero the whole axis is selected. Negative values count
// from the end of the axis.
type Range struct {
	Start int
	Stop  int
}

// R is a convenience constructor for a Range.
func R(start, stop int) Range { return Range{Start: start, Stop: stop} }

func (r Range) resolve(n int) (int, int) {
	start, stop := r.Start, r.Stop
	if start == 0 && stop == 0 {
		return 0, n
	}
	if start < 0 {
		start += n
	}
	if stop < 0 {
		stop += n
	}
	if start < 0 {
		start = 0
	}
	if stop > n {
		stop = n
	}
	if stop < start {
		stop = start
	}
	return start, stop
}

// Concatenate joins arrays along an existing axis. All arrays must have the
// same shape except along the concatenation axis.
func Concatenate(axis int, arrays ...*NDArray) *NDArray {
	if len(arrays) == 0 {
		panic("numpy: Concatenate needs at least one array")
	}
	first := arrays[0]
	if axis < 0 {
		axis += first.ndim
	}
	if axis < 0 || axis >= first.ndim {
		panic(fmt.Sprintf("numpy: Concatenate axis %d out of range for ndim %d", axis, first.ndim))
	}
	outShape := first.Shape()
	axisTotal := 0
	for _, arr := range arrays {
		if arr.ndim != first.ndim {
			panic("numpy: Concatenate arrays must share ndim")
		}
		for d := 0; d < first.ndim; d++ {
			if d == axis {
				continue
			}
			if arr.shape[d] != first.shape[d] {
				panic(fmt.Sprintf("numpy: Concatenate shape mismatch on axis %d", d))
			}
		}
		axisTotal += arr.shape[axis]
	}
	outShape[axis] = axisTotal
	out := Zeros(outShape...)
	axisOffset := 0
	for _, arr := range arrays {
		arr.forEach(func(off int, idx []int) {
			outIdx := make([]int, len(idx))
			copy(outIdx, idx)
			outIdx[axis] += axisOffset
			out.data[out.offset(outIdx)] = arr.data[off]
		})
		axisOffset += arr.shape[axis]
	}
	return out
}

// Stack joins arrays along a new axis. All arrays must have identical shapes.
// The result has one more dimension than the inputs, inserted at axis.
func Stack(axis int, arrays ...*NDArray) *NDArray {
	if len(arrays) == 0 {
		panic("numpy: Stack needs at least one array")
	}
	first := arrays[0]
	newNdim := first.ndim + 1
	if axis < 0 {
		axis += newNdim
	}
	if axis < 0 || axis >= newNdim {
		panic(fmt.Sprintf("numpy: Stack axis %d out of range", axis))
	}
	for _, arr := range arrays {
		if arr.ndim != first.ndim {
			panic("numpy: Stack arrays must share shape")
		}
		for d := range first.shape {
			if arr.shape[d] != first.shape[d] {
				panic("numpy: Stack arrays must share shape")
			}
		}
	}
	outShape := make([]int, newNdim)
	j := 0
	for i := 0; i < newNdim; i++ {
		if i == axis {
			outShape[i] = len(arrays)
		} else {
			outShape[i] = first.shape[j]
			j++
		}
	}
	out := Zeros(outShape...)
	for k, arr := range arrays {
		arr.forEach(func(off int, idx []int) {
			outIdx := make([]int, newNdim)
			jj := 0
			for i := 0; i < newNdim; i++ {
				if i == axis {
					outIdx[i] = k
				} else {
					outIdx[i] = idx[jj]
					jj++
				}
			}
			out.data[out.offset(outIdx)] = arr.data[off]
		})
	}
	return out
}
