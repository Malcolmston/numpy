package numpy

import (
	"math"
	"testing"
)

func approxSlice(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range got {
		if math.IsNaN(want[i]) {
			if !math.IsNaN(got[i]) {
				t.Fatalf("index %d: got %v want NaN", i, got[i])
			}
			continue
		}
		if math.Abs(got[i]-want[i]) > tol {
			t.Fatalf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestUnaryMathExtra(t *testing.T) {
	src := FromSlice([]float64{0, 0.5, 1, 2, 4})
	tests := []struct {
		name string
		got  []float64
		want []float64
	}{
		{"Tanh", FromSlice([]float64{0, 1, -1}).Tanh().Data(), []float64{0, math.Tanh(1), math.Tanh(-1)}},
		{"Sinh", FromSlice([]float64{0, 1}).Sinh().Data(), []float64{0, math.Sinh(1)}},
		{"Cosh", FromSlice([]float64{0, 1}).Cosh().Data(), []float64{1, math.Cosh(1)}},
		{"Tan", FromSlice([]float64{0}).Tan().Data(), []float64{0}},
		{"Arcsin", FromSlice([]float64{0, 1}).Arcsin().Data(), []float64{0, math.Pi / 2}},
		{"Arccos", FromSlice([]float64{1, 0}).Arccos().Data(), []float64{0, math.Pi / 2}},
		{"Arctan", FromSlice([]float64{0, 1}).Arctan().Data(), []float64{0, math.Pi / 4}},
		{"Square", src.Square().Data(), []float64{0, 0.25, 1, 4, 16}},
		{"Reciprocal", FromSlice([]float64{1, 2, 4}).Reciprocal().Data(), []float64{1, 0.5, 0.25}},
		{"Cbrt", FromSlice([]float64{8, 27}).Cbrt().Data(), []float64{2, 3}},
		{"Log2", FromSlice([]float64{1, 2, 8}).Log2().Data(), []float64{0, 1, 3}},
		{"Log10", FromSlice([]float64{1, 10, 100}).Log10().Data(), []float64{0, 1, 2}},
		{"Log1p", FromSlice([]float64{0}).Log1p().Data(), []float64{0}},
		{"Expm1", FromSlice([]float64{0}).Expm1().Data(), []float64{0}},
		{"Floor", FromSlice([]float64{1.7, -1.2}).Floor().Data(), []float64{1, -2}},
		{"Ceil", FromSlice([]float64{1.2, -1.7}).Ceil().Data(), []float64{2, -1}},
		{"Round", FromSlice([]float64{0.5, 1.5, 2.5, 2.6}).Round().Data(), []float64{0, 2, 2, 3}},
		{"Trunc", FromSlice([]float64{1.7, -1.7}).Trunc().Data(), []float64{1, -1}},
		{"Sign", FromSlice([]float64{-3, 0, 4}).Sign().Data(), []float64{-1, 0, 1}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			approxSlice(t, tc.got, tc.want, 1e-12)
		})
	}
}

func TestSignNaN(t *testing.T) {
	got := FromSlice([]float64{math.NaN()}).Sign().Data()
	if !math.IsNaN(got[0]) {
		t.Fatalf("Sign(NaN) = %v, want NaN", got[0])
	}
}

func TestBinaryMathExtra(t *testing.T) {
	y := FromSlice([]float64{1, 1, -1})
	x := FromSlice([]float64{1, -1, -1})
	approxSlice(t, y.Arctan2(x).Data(),
		[]float64{math.Pi / 4, 3 * math.Pi / 4, -3 * math.Pi / 4}, 1e-12)

	a := FromSlice([]float64{3, 5})
	b := FromSlice([]float64{4, 12})
	approxSlice(t, a.Hypot(b).Data(), []float64{5, 13}, 1e-12)

	m := FromSlice([]float64{7, -7, 8})
	n := FromSlice([]float64{3, 3, 5})
	approxSlice(t, m.Mod(n).Data(), []float64{1, -1, 3}, 1e-12)
}

func TestClip(t *testing.T) {
	got := FromSlice([]float64{-2, -1, 0, 1, 2, 3}).Clip(-1, 2).Data()
	approxSlice(t, got, []float64{-1, -1, 0, 1, 2, 2}, 0)
}

func TestClipPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for min > max")
		}
	}()
	FromSlice([]float64{1}).Clip(2, 1)
}

func BenchmarkTanh(b *testing.B) {
	a := Arange(0, 1000, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Tanh()
	}
}
