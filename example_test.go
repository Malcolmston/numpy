package numpy_test

import (
	"fmt"

	np "github.com/malcolmston/numpy"
)

func Example() {
	// Build a 2x3 array with Arange + Reshape.
	a := np.Arange(0, 6, 1).Reshape(2, 3)

	// Broadcast-add a row vector of shape (3,).
	b := np.FromSlice([]float64{10, 20, 30})
	sum := a.Add(b)

	// Reduce along axis 0 (down the columns).
	cols := a.SumAxis(0, false)

	// Matrix multiply a (2x3) with its transpose (3x2) -> (2x2).
	gram := a.MatMul(a.T())

	fmt.Println("a      =", a.Data())
	fmt.Println("a+b    =", sum.Data())
	fmt.Println("colsum =", cols.Data())
	fmt.Println("gram   =", gram.Data())
	fmt.Println("mean   =", a.Mean())
	// Output:
	// a      = [0 1 2 3 4 5]
	// a+b    = [10 21 32 13 24 35]
	// colsum = [3 5 7]
	// gram   = [5 14 14 50]
	// mean   = 2.5
}

func ExampleNDArray_MaskSelect() {
	a := np.Arange(0, 6, 1).Reshape(2, 3)
	mask := a.GreaterScalar(2)
	fmt.Println(a.MaskSelect(mask).Data())
	// Output:
	// [3 4 5]
}
