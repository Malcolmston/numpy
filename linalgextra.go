package numpy

import (
	"fmt"
	"math"
)

// Trace returns the sum of the elements on the main diagonal of a 2-D array.
// It panics if a is not 2-D.
func (a *NDArray) Trace() float64 {
	if a.ndim != 2 {
		panic(fmt.Sprintf("numpy: Trace requires a 2-D array, got %dD", a.ndim))
	}
	n := a.shape[0]
	if a.shape[1] < n {
		n = a.shape[1]
	}
	var s float64
	for i := 0; i < n; i++ {
		s += a.data[i*a.strides[0]+i*a.strides[1]]
	}
	return s
}

// Diagonal returns the k-th diagonal of a 2-D array as a new 1-D array. A
// positive k selects a diagonal above the main one, a negative k below it. It
// panics if a is not 2-D.
func (a *NDArray) Diagonal(k int) *NDArray {
	if a.ndim != 2 {
		panic(fmt.Sprintf("numpy: Diagonal requires a 2-D array, got %dD", a.ndim))
	}
	rows, cols := a.shape[0], a.shape[1]
	out := make([]float64, 0, rows)
	for i := 0; i < rows; i++ {
		j := i + k
		if j >= 0 && j < cols {
			out = append(out, a.data[i*a.strides[0]+j*a.strides[1]])
		}
	}
	return newArray(out, []int{len(out)})
}

// Outer returns the outer product of two arrays. Both operands are flattened to
// 1-D; for inputs of length m and n the result is an (m, n) matrix with
// out[i,j] = a[i] * b[j].
func (a *NDArray) Outer(b *NDArray) *NDArray {
	av := a.Data()
	bv := b.Data()
	out := Zeros(len(av), len(bv))
	for i, x := range av {
		row := i * len(bv)
		for j, y := range bv {
			out.data[row+j] = x * y
		}
	}
	return out
}

// Cross returns the cross product of two 1-D arrays, mirroring NumPy's
// numpy.cross for 1-D inputs. Each operand must have length 2 or 3, with any
// missing third component treated as zero:
//
//   - two 2-vectors yield the scalar z-component a[0]*b[1] - a[1]*b[0],
//     returned as a 0-dimensional array whose single value is the result;
//   - any combination involving a 3-vector yields a 3-vector.
//
// It panics if either operand is not 1-D or has a length other than 2 or 3.
func (a *NDArray) Cross(b *NDArray) *NDArray {
	if a.ndim != 1 || b.ndim != 1 {
		panic("numpy: Cross requires two 1-D arrays")
	}
	na, nb := a.shape[0], b.shape[0]
	if (na != 2 && na != 3) || (nb != 2 && nb != 3) {
		panic("numpy: Cross requires vectors of length 2 or 3")
	}
	av := a.Data()
	bv := b.Data()
	// Pad each operand to length 3 with a zero third component.
	a0, a1, a2 := av[0], av[1], 0.0
	if na == 3 {
		a2 = av[2]
	}
	b0, b1, b2 := bv[0], bv[1], 0.0
	if nb == 3 {
		b2 = bv[2]
	}
	if na == 2 && nb == 2 {
		// Both 2-vectors: NumPy returns the scalar z-component as a
		// 0-dimensional array.
		return newArray([]float64{a0*b1 - a1*b0}, []int{})
	}
	out := []float64{
		a1*b2 - a2*b1,
		a2*b0 - a0*b2,
		a0*b1 - a1*b0,
	}
	return newArray(out, []int{3})
}

// Norm returns the Frobenius (L2) norm of a: the square root of the sum of the
// squares of all elements. For a 1-D array this is the Euclidean vector norm.
func (a *NDArray) Norm() float64 {
	var s float64
	a.forEach(func(off int, _ []int) {
		v := a.data[off]
		s += v * v
	})
	return math.Sqrt(s)
}

// Diag has two behaviors mirroring NumPy. Given a 1-D array of length n it
// returns the n-by-n matrix with that array on its main diagonal and zeros
// elsewhere. Given a 2-D array it returns its main diagonal as a new 1-D array.
// It panics for arrays of other dimensionality.
func Diag(a *NDArray) *NDArray {
	switch a.ndim {
	case 1:
		n := a.shape[0]
		out := Zeros(n, n)
		v := a.Data()
		for i := 0; i < n; i++ {
			out.data[i*n+i] = v[i]
		}
		return out
	case 2:
		return a.Diagonal(0)
	default:
		panic(fmt.Sprintf("numpy: Diag requires a 1-D or 2-D array, got %dD", a.ndim))
	}
}
