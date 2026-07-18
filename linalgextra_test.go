package numpy

import (
	"math"
	"reflect"
	"testing"
)

func TestTrace(t *testing.T) {
	a := FromData([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	if got := a.Trace(); got != 15 {
		t.Fatalf("Trace = %v want 15", got)
	}
	// Non-square: sum of min(m,n) diagonal entries.
	b := FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if got := b.Trace(); got != 6 { // 1 + 5
		t.Fatalf("Trace non-square = %v want 6", got)
	}
}

func TestDiagonal(t *testing.T) {
	a := FromData([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	if got := a.Diagonal(0).Data(); !reflect.DeepEqual(got, []float64{1, 5, 9}) {
		t.Fatalf("Diagonal(0) = %v", got)
	}
	if got := a.Diagonal(1).Data(); !reflect.DeepEqual(got, []float64{2, 6}) {
		t.Fatalf("Diagonal(1) = %v", got)
	}
	if got := a.Diagonal(-1).Data(); !reflect.DeepEqual(got, []float64{4, 8}) {
		t.Fatalf("Diagonal(-1) = %v", got)
	}
}

func TestOuter(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	b := FromSlice([]float64{4, 5})
	got := a.Outer(b)
	if !reflect.DeepEqual(got.Shape(), []int{3, 2}) {
		t.Fatalf("Outer shape = %v", got.Shape())
	}
	want := []float64{4, 5, 8, 10, 12, 15}
	if !reflect.DeepEqual(got.Data(), want) {
		t.Fatalf("Outer = %v want %v", got.Data(), want)
	}
}

func TestCross(t *testing.T) {
	x := FromSlice([]float64{1, 0, 0})
	y := FromSlice([]float64{0, 1, 0})
	got := x.Cross(y).Data()
	if !reflect.DeepEqual(got, []float64{0, 0, 1}) {
		t.Fatalf("Cross = %v want [0 0 1]", got)
	}
}

func TestNorm(t *testing.T) {
	a := FromSlice([]float64{3, 4})
	if got := a.Norm(); math.Abs(got-5) > 1e-12 {
		t.Fatalf("Norm = %v want 5", got)
	}
	m := FromData([]float64{1, 2, 3, 4}, 2, 2)
	if got := m.Norm(); math.Abs(got-math.Sqrt(30)) > 1e-12 {
		t.Fatalf("Norm 2D = %v want %v", got, math.Sqrt(30))
	}
}

func TestDiag(t *testing.T) {
	v := FromSlice([]float64{1, 2, 3})
	m := Diag(v)
	if !reflect.DeepEqual(m.Shape(), []int{3, 3}) {
		t.Fatalf("Diag shape = %v", m.Shape())
	}
	want := []float64{1, 0, 0, 0, 2, 0, 0, 0, 3}
	if !reflect.DeepEqual(m.Data(), want) {
		t.Fatalf("Diag = %v want %v", m.Data(), want)
	}
	// Round trip: extracting the diagonal returns the original vector.
	if got := Diag(m).Data(); !reflect.DeepEqual(got, []float64{1, 2, 3}) {
		t.Fatalf("Diag round trip = %v", got)
	}
}

func BenchmarkOuter(b *testing.B) {
	x := Arange(0, 100, 1)
	y := Arange(0, 100, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = x.Outer(y)
	}
}
