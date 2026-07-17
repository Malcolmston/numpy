package numpy

import "fmt"

// MatMul returns the matrix product of two 2-D arrays. The receiver has shape
// (m, k) and b has shape (k, n); the result has shape (m, n). It panics on a
// dimension mismatch or non-2-D input.
func (a *NDArray) MatMul(b *NDArray) *NDArray {
	if a.ndim != 2 || b.ndim != 2 {
		panic(fmt.Sprintf("numpy: MatMul requires 2-D arrays, got %dD and %dD", a.ndim, b.ndim))
	}
	m, k := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]
	if k != k2 {
		panic(fmt.Sprintf("numpy: MatMul inner dimensions %d and %d do not match", k, k2))
	}
	out := Zeros(m, n)
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			aip := a.data[i*a.strides[0]+p*a.strides[1]]
			if aip == 0 {
				continue
			}
			for j := 0; j < n; j++ {
				out.data[i*n+j] += aip * b.data[p*b.strides[0]+j*b.strides[1]]
			}
		}
	}
	return out
}

// Dot computes a dot product. For two 1-D arrays it returns a scalar array of
// shape (1). For two 2-D arrays it is equivalent to MatMul. Other combinations
// panic.
func (a *NDArray) Dot(b *NDArray) *NDArray {
	switch {
	case a.ndim == 1 && b.ndim == 1:
		if a.shape[0] != b.shape[0] {
			panic(fmt.Sprintf("numpy: Dot length mismatch %d and %d", a.shape[0], b.shape[0]))
		}
		var s float64
		for i := 0; i < a.shape[0]; i++ {
			s += a.data[i*a.strides[0]] * b.data[i*b.strides[0]]
		}
		return newArray([]float64{s}, []int{1})
	case a.ndim == 2 && b.ndim == 2:
		return a.MatMul(b)
	default:
		panic(fmt.Sprintf("numpy: Dot supports 1-D*1-D or 2-D*2-D, got %dD and %dD", a.ndim, b.ndim))
	}
}
