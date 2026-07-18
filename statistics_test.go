package numpy

import (
	"math"
	"testing"
)

func TestMedian(t *testing.T) {
	if got := FromSlice([]float64{1, 3, 2}).Median(); got != 2 {
		t.Fatalf("odd Median = %v want 2", got)
	}
	if got := FromSlice([]float64{1, 2, 3, 4}).Median(); got != 2.5 {
		t.Fatalf("even Median = %v want 2.5", got)
	}
}

func TestPercentile(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4})
	cases := []struct {
		q    float64
		want float64
	}{
		{0, 0}, {50, 2}, {100, 4}, {25, 1}, {75, 3}, {10, 0.4},
	}
	for _, c := range cases {
		if got := a.Percentile(c.q); math.Abs(got-c.want) > 1e-12 {
			t.Fatalf("Percentile(%v) = %v want %v", c.q, got, c.want)
		}
	}
}

func TestQuantile(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4})
	if got := a.Quantile(0.5); got != 2 {
		t.Fatalf("Quantile(0.5) = %v want 2", got)
	}
}

func TestPercentilePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for q out of range")
		}
	}()
	FromSlice([]float64{1}).Percentile(150)
}

func TestVarStdDDof(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5})
	// population variance = 2, sample variance (ddof=1) = 2.5.
	if got := a.VarDDof(0); math.Abs(got-2) > 1e-12 {
		t.Fatalf("VarDDof(0) = %v want 2", got)
	}
	if got := a.VarDDof(1); math.Abs(got-2.5) > 1e-12 {
		t.Fatalf("VarDDof(1) = %v want 2.5", got)
	}
	if got := a.StdDDof(1); math.Abs(got-math.Sqrt(2.5)) > 1e-12 {
		t.Fatalf("StdDDof(1) = %v want %v", got, math.Sqrt(2.5))
	}
	// Consistency with the existing population Var.
	if math.Abs(a.VarDDof(0)-a.Var()) > 1e-12 {
		t.Fatalf("VarDDof(0) inconsistent with Var")
	}
}

func TestVarDDofPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for N-ddof <= 0")
		}
	}()
	FromSlice([]float64{1}).VarDDof(1)
}
