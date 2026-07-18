package numpy

import (
	"math"
	"testing"
)

// The tests in this file encode concrete known-answer vectors taken directly
// from the upstream NumPy test suite (numpy/numpy, tag v1.26.4) so that the Go
// port's behaviour can be checked against the values the original library
// asserts. Each test cites the upstream source file and class it mirrors.

// approx reports whether every element of got matches want within tol.
func parityApprox(got, want []float64, tol float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			return false
		}
	}
	return true
}

// TestParityCross mirrors numpy/core/tests/test_numeric.py::TestCross
// (test_2x2, test_2x3, test_3x3). NumPy defines the cross product for 1-D
// vectors of length 2 or 3, padding a missing third component with zero.
func TestParityCross(t *testing.T) {
	// test_2x2: cross([1,2],[3,4]) == -2 (scalar), and the reverse is +2.
	u2 := FromSlice([]float64{1, 2})
	v2 := FromSlice([]float64{3, 4})
	if got := u2.Cross(v2); got.Ndim() != 0 || got.Data()[0] != -2 {
		t.Fatalf("cross([1,2],[3,4]) = %v (ndim %d), want scalar -2", got.Data(), got.Ndim())
	}
	if got := v2.Cross(u2); got.Data()[0] != 2 {
		t.Fatalf("cross([3,4],[1,2]) = %v, want 2", got.Data())
	}

	// test_2x3: cross([1,2],[3,4,5]) == [10,-5,-2]; reverse negates.
	u23 := FromSlice([]float64{1, 2})
	v23 := FromSlice([]float64{3, 4, 5})
	if got := u23.Cross(v23).Data(); !parityApprox(got, []float64{10, -5, -2}, 0) {
		t.Fatalf("cross([1,2],[3,4,5]) = %v, want [10 -5 -2]", got)
	}
	if got := v23.Cross(u23).Data(); !parityApprox(got, []float64{-10, 5, 2}, 0) {
		t.Fatalf("cross([3,4,5],[1,2]) = %v, want [-10 5 2]", got)
	}

	// test_3x3: cross([1,2,3],[4,5,6]) == [-3,6,-3]; reverse negates.
	u3 := FromSlice([]float64{1, 2, 3})
	v3 := FromSlice([]float64{4, 5, 6})
	if got := u3.Cross(v3).Data(); !parityApprox(got, []float64{-3, 6, -3}, 0) {
		t.Fatalf("cross([1,2,3],[4,5,6]) = %v, want [-3 6 -3]", got)
	}
	if got := v3.Cross(u3).Data(); !parityApprox(got, []float64{3, -6, 3}, 0) {
		t.Fatalf("cross([4,5,6],[1,2,3]) = %v, want [3 -6 3]", got)
	}

	// 3x2 mixed case, from the broadcasting section: cross([1,2,3],[3,4]) ==
	// [-12,9,-2].
	if got := FromSlice([]float64{1, 2, 3}).Cross(FromSlice([]float64{3, 4})).Data(); !parityApprox(got, []float64{-12, 9, -2}, 0) {
		t.Fatalf("cross([1,2,3],[3,4]) = %v, want [-12 9 -2]", got)
	}

	// cross(u,u) == 0 for a 2-vector.
	if got := u2.Cross(u2).Data()[0]; got != 0 {
		t.Fatalf("cross(u,u) = %v, want 0", got)
	}
}

// TestParityDot mirrors the classic 1-D inner product and 2-D matrix product
// checks in numpy/core/tests/test_multiarray.py::TestDot / TestMatmul.
func TestParityDot(t *testing.T) {
	// dot([1,2,3],[4,5,6]) == 32.
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{4, 5, 6})
	if got := a.Dot(b).Data()[0]; got != 32 {
		t.Fatalf("dot 1-D = %v, want 32", got)
	}
	// [[1,2],[3,4]] @ [[1,2],[3,4]] == [[7,10],[15,22]].
	m := FromData([]float64{1, 2, 3, 4}, 2, 2)
	if got := m.MatMul(m).Data(); !parityApprox(got, []float64{7, 10, 15, 22}, 0) {
		t.Fatalf("matmul = %v, want [7 10 15 22]", got)
	}
}

// TestParityOuter mirrors numpy/core/tests/test_multiarray.py::TestOuter.
// np.outer([1,2,3],[4,5]) == [[4,5],[8,10],[12,15]].
func TestParityOuter(t *testing.T) {
	got := FromSlice([]float64{1, 2, 3}).Outer(FromSlice([]float64{4, 5}))
	if s := got.Shape(); len(s) != 2 || s[0] != 3 || s[1] != 2 {
		t.Fatalf("outer shape = %v, want [3 2]", s)
	}
	if !parityApprox(got.Data(), []float64{4, 5, 8, 10, 12, 15}, 0) {
		t.Fatalf("outer = %v", got.Data())
	}
}

// TestParityTraceDiagonal mirrors the trace/diagonal identities checked in
// numpy/core/tests/test_multiarray.py (TestMethods) on the 3x3 range matrix.
func TestParityTraceDiagonal(t *testing.T) {
	m := Arange(0, 9, 1).Reshape(3, 3) // [[0,1,2],[3,4,5],[6,7,8]]
	if got := m.Trace(); got != 12 {   // 0+4+8
		t.Fatalf("trace = %v, want 12", got)
	}
	if got := m.Diagonal(0).Data(); !parityApprox(got, []float64{0, 4, 8}, 0) {
		t.Fatalf("diagonal(0) = %v", got)
	}
	if got := m.Diagonal(1).Data(); !parityApprox(got, []float64{1, 5}, 0) {
		t.Fatalf("diagonal(1) = %v, want [1 5]", got)
	}
	if got := m.Diagonal(-1).Data(); !parityApprox(got, []float64{3, 7}, 0) {
		t.Fatalf("diagonal(-1) = %v, want [3 7]", got)
	}
}

// TestParityArange mirrors numpy/core/tests/test_multiarray.py::TestArange /
// the documented behaviour: arange(0,6) == [0,1,2,3,4,5], and a fractional
// step is honoured.
func TestParityArange(t *testing.T) {
	if got := Arange(0, 6, 1).Data(); !parityApprox(got, []float64{0, 1, 2, 3, 4, 5}, 0) {
		t.Fatalf("arange(0,6) = %v", got)
	}
	if got := Arange(2, 3, 0.1).Data(); len(got) != 10 || math.Abs(got[9]-2.9) > 1e-9 {
		t.Fatalf("arange(2,3,0.1) = %v (len %d)", got, len(got))
	}
}

// TestParityLinspace mirrors numpy/core/tests/test_function_base.py::TestLinspace.
// linspace(0,1,5) == [0,0.25,0.5,0.75,1]; endpoint=False drops the endpoint.
func TestParityLinspace(t *testing.T) {
	if got := Linspace(0, 1, 5, true).Data(); !parityApprox(got, []float64{0, 0.25, 0.5, 0.75, 1}, 1e-12) {
		t.Fatalf("linspace endpoint = %v", got)
	}
	if got := Linspace(0, 1, 5, false).Data(); !parityApprox(got, []float64{0, 0.2, 0.4, 0.6, 0.8}, 1e-12) {
		t.Fatalf("linspace no-endpoint = %v", got)
	}
	// A single sample returns start (upstream TestLinspace.test_corner).
	if got := Linspace(0, 1, 1, true).Data(); len(got) != 1 || got[0] != 0 {
		t.Fatalf("linspace num=1 = %v", got)
	}
}

// TestParityCumsumProd mirrors numpy/core/tests/test_multiarray.py cumsum/
// cumprod checks. cumsum([1,2,3,4]) == [1,3,6,10]; cumprod == [1,2,6,24].
func TestParityCumsumProd(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4})
	if got := a.Cumsum().Data(); !parityApprox(got, []float64{1, 3, 6, 10}, 0) {
		t.Fatalf("cumsum = %v", got)
	}
	if got := a.Cumprod().Data(); !parityApprox(got, []float64{1, 2, 6, 24}, 0) {
		t.Fatalf("cumprod = %v", got)
	}
}

// TestParityDiff mirrors numpy/lib/tests/test_function_base.py::TestDiff.
// diff([1,4,9,16]) == [3,5,7].
func TestParityDiff(t *testing.T) {
	if got := FromSlice([]float64{1, 4, 9, 16}).Diff().Data(); !parityApprox(got, []float64{3, 5, 7}, 0) {
		t.Fatalf("diff = %v", got)
	}
}

// TestParityMedianPercentile mirrors numpy/lib/tests/test_function_base.py
// (TestMedian, TestPercentile). median([1,2,3,4]) == 2.5; the default "linear"
// percentile of that data at q=25 is 1.75, q=50 is 2.5, q=75 is 3.25.
func TestParityMedianPercentile(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4})
	if got := a.Median(); got != 2.5 {
		t.Fatalf("median = %v, want 2.5", got)
	}
	if got := a.Percentile(25); math.Abs(got-1.75) > 1e-12 {
		t.Fatalf("percentile(25) = %v, want 1.75", got)
	}
	if got := a.Percentile(50); math.Abs(got-2.5) > 1e-12 {
		t.Fatalf("percentile(50) = %v, want 2.5", got)
	}
	if got := a.Percentile(75); math.Abs(got-3.25) > 1e-12 {
		t.Fatalf("percentile(75) = %v, want 3.25", got)
	}
	// median of an odd-length list (TestMedian.test_basic): median([1,2,3]) == 2.
	if got := FromSlice([]float64{1, 2, 3}).Median(); got != 2 {
		t.Fatalf("median odd = %v, want 2", got)
	}
}

// TestParityVarStd mirrors numpy/core/tests/test_multiarray.py var/std checks.
// For [1,2,3,4] the population variance (ddof=0) is 1.25 and std is sqrt(1.25);
// the sample variance (ddof=1) is 5/3.
func TestParityVarStd(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4})
	if got := a.Var(); math.Abs(got-1.25) > 1e-12 {
		t.Fatalf("var = %v, want 1.25", got)
	}
	if got := a.Std(); math.Abs(got-math.Sqrt(1.25)) > 1e-12 {
		t.Fatalf("std = %v", got)
	}
	if got := a.VarDDof(1); math.Abs(got-5.0/3.0) > 1e-12 {
		t.Fatalf("var ddof=1 = %v, want %v", got, 5.0/3.0)
	}
}

// TestParitySortArgsort mirrors numpy/core/tests/test_multiarray.py::TestSort.
// sort([3,1,2]) == [1,2,3]; argsort([3,1,2]) == [1,2,0].
func TestParitySortArgsort(t *testing.T) {
	a := FromSlice([]float64{3, 1, 2})
	if got := a.Sort().Data(); !parityApprox(got, []float64{1, 2, 3}, 0) {
		t.Fatalf("sort = %v", got)
	}
	if got := a.Argsort().Data(); !parityApprox(got, []float64{1, 2, 0}, 0) {
		t.Fatalf("argsort = %v, want [1 2 0]", got)
	}
}

// TestParitySearchSortedUnique mirrors numpy searchsorted/unique documented
// vectors. searchsorted([1,2,3,4,5],3) == 2; unique([1,1,2,2,3]) == [1,2,3].
func TestParitySearchSortedUnique(t *testing.T) {
	if got := FromSlice([]float64{1, 2, 3, 4, 5}).SearchSorted(3); got != 2 {
		t.Fatalf("searchsorted = %v, want 2", got)
	}
	if got := FromSlice([]float64{1, 1, 2, 2, 3}).Unique().Data(); !parityApprox(got, []float64{1, 2, 3}, 0) {
		t.Fatalf("unique = %v", got)
	}
}

// TestParityArgmaxArgmin mirrors numpy/core/tests/test_multiarray.py argmax/
// argmin: for [1,3,2,3] argmax returns the first max (index 1) and argmin the
// first min (index 0).
func TestParityArgmaxArgmin(t *testing.T) {
	a := FromSlice([]float64{1, 3, 2, 3})
	if got := a.Argmax(); got != 1 {
		t.Fatalf("argmax = %v, want 1", got)
	}
	if got := a.Argmin(); got != 0 {
		t.Fatalf("argmin = %v, want 0", got)
	}
}

// TestParityClip mirrors numpy/core/tests/test_numeric.py::TestClip.
// clip([-2,-1,0,1,2], -1, 1) == [-1,-1,0,1,1].
func TestParityClip(t *testing.T) {
	got := FromSlice([]float64{-2, -1, 0, 1, 2}).Clip(-1, 1).Data()
	if !parityApprox(got, []float64{-1, -1, 0, 1, 1}, 0) {
		t.Fatalf("clip = %v", got)
	}
}

// TestParityRollFlip mirrors numpy roll/flip documented vectors.
// roll([0,1,2,3,4],2) == [3,4,0,1,2]; flip([0,1,2,3,4]) == [4,3,2,1,0].
func TestParityRollFlip(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4})
	if got := a.Roll(2).Data(); !parityApprox(got, []float64{3, 4, 0, 1, 2}, 0) {
		t.Fatalf("roll = %v", got)
	}
	if got := a.Flip(0).Data(); !parityApprox(got, []float64{4, 3, 2, 1, 0}, 0) {
		t.Fatalf("flip = %v", got)
	}
}

// TestParitySignRound mirrors numpy/core/tests/test_umath.py sign and rint.
// sign([-5,0,3]) == [-1,0,1]; round uses banker's rounding, so
// round([0.5,1.5,2.5,3.5]) == [0,2,2,4].
func TestParitySignRound(t *testing.T) {
	if got := FromSlice([]float64{-5, 0, 3}).Sign().Data(); !parityApprox(got, []float64{-1, 0, 1}, 0) {
		t.Fatalf("sign = %v", got)
	}
	if got := FromSlice([]float64{0.5, 1.5, 2.5, 3.5}).Round().Data(); !parityApprox(got, []float64{0, 2, 2, 4}, 0) {
		t.Fatalf("round (banker's) = %v, want [0 2 2 4]", got)
	}
}

// TestParityBroadcastAdd mirrors the canonical broadcasting example from the
// NumPy documentation and quick-start: a (2,3) matrix plus a (3,) row vector.
// arange(0,6).reshape(2,3) + [10,20,30] == [[10,21,32],[13,24,35]].
func TestParityBroadcastAdd(t *testing.T) {
	a := Arange(0, 6, 1).Reshape(2, 3)
	b := FromSlice([]float64{10, 20, 30})
	if got := a.Add(b).Data(); !parityApprox(got, []float64{10, 21, 32, 13, 24, 35}, 0) {
		t.Fatalf("broadcast add = %v", got)
	}
	// Reduction along axis 0 (TestMethods sum-axis vectors): sum == [3,5,7].
	if got := a.SumAxis(0, false).Data(); !parityApprox(got, []float64{3, 5, 7}, 0) {
		t.Fatalf("sum axis0 = %v", got)
	}
}

// TestParityCrossGram mirrors the a @ a.T Gram-matrix example: for
// arange(0,6).reshape(2,3), a @ a.T == [[5,14],[14,50]].
func TestParityGram(t *testing.T) {
	a := Arange(0, 6, 1).Reshape(2, 3)
	if got := a.MatMul(a.T()).Data(); !parityApprox(got, []float64{5, 14, 14, 50}, 0) {
		t.Fatalf("gram = %v", got)
	}
}
