package numpy

import "fmt"

// broadcastShape computes the broadcast result shape of two shapes following
// numpy rules: aligned from the trailing axis, dimensions must be equal or one
// of them must be 1. It panics on incompatible shapes.
func broadcastShape(a, b []int) []int {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		da, db := 1, 1
		if i < len(a) {
			da = a[len(a)-1-i]
		}
		if i < len(b) {
			db = b[len(b)-1-i]
		}
		switch {
		case da == db:
			out[n-1-i] = da
		case da == 1:
			out[n-1-i] = db
		case db == 1:
			out[n-1-i] = da
		default:
			panic(fmt.Sprintf("numpy: shapes %v and %v are not broadcast-compatible", a, b))
		}
	}
	return out
}

// broadcastStrides returns strides that view a (with shape ashape) as if it had
// shape target, using zero strides for broadcast axes. It panics if a cannot be
// broadcast to target.
func broadcastStrides(strides, ashape, target []int) []int {
	out := make([]int, len(target))
	offset := len(target) - len(ashape)
	for i := range target {
		src := i - offset
		if src < 0 {
			out[i] = 0
			continue
		}
		switch ashape[src] {
		case target[i]:
			out[i] = strides[src]
		case 1:
			out[i] = 0
		default:
			panic(fmt.Sprintf("numpy: cannot broadcast shape %v to %v", ashape, target))
		}
	}
	return out
}

// broadcastView returns a zero-copy view of a with the given target shape.
func (a *NDArray) broadcastView(target []int) *NDArray {
	strides := broadcastStrides(a.strides, a.shape, target)
	shape := make([]int, len(target))
	copy(shape, target)
	return &NDArray{
		data:    a.data,
		shape:   shape,
		strides: strides,
		ndim:    len(shape),
		size:    sizeOf(shape),
	}
}

// BroadcastTo returns a contiguous copy of a broadcast to shape.
func (a *NDArray) BroadcastTo(shape ...int) *NDArray {
	return a.broadcastView(shape).Copy()
}

// binaryOp applies fn element-wise to a and b with broadcasting, returning a
// new contiguous array.
func binaryOp(a, b *NDArray, fn func(x, y float64) float64) *NDArray {
	target := broadcastShape(a.shape, b.shape)
	av := a.broadcastView(target)
	bv := b.broadcastView(target)
	out := Zeros(target...)
	i := 0
	// Both views share the same shape, so a single walk drives all three.
	bOff := make([]int, out.size)
	j := 0
	bv.forEach(func(off int, _ []int) {
		bOff[j] = off
		j++
	})
	av.forEach(func(off int, _ []int) {
		out.data[i] = fn(av.data[off], bv.data[bOff[i]])
		i++
	})
	return out
}

// unaryOp applies fn element-wise to a, returning a new contiguous array.
func unaryOp(a *NDArray, fn func(x float64) float64) *NDArray {
	out := Zeros(a.shape...)
	i := 0
	a.forEach(func(off int, _ []int) {
		out.data[i] = fn(a.data[off])
		i++
	})
	return out
}
