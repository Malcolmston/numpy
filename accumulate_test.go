package numpy

import (
	"reflect"
	"testing"
)

func TestCumsum(t *testing.T) {
	a := FromData([]float64{1, 2, 3, 4}, 2, 2)
	got := a.Cumsum().Data()
	want := []float64{1, 3, 6, 10}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Cumsum = %v want %v", got, want)
	}
}

func TestCumprod(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4})
	got := a.Cumprod().Data()
	want := []float64{1, 2, 6, 24}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Cumprod = %v want %v", got, want)
	}
}

func TestDiff(t *testing.T) {
	a := FromSlice([]float64{1, 2, 4, 7, 0})
	got := a.Diff().Data()
	want := []float64{1, 2, 3, -7}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Diff = %v want %v", got, want)
	}
	// Degenerate cases.
	if n := FromSlice([]float64{5}).Diff().Size(); n != 0 {
		t.Fatalf("Diff single element size = %d want 0", n)
	}
}

func TestPtp(t *testing.T) {
	a := FromData([]float64{3, 1, 9, 4}, 2, 2)
	if got := a.Ptp(); got != 8 {
		t.Fatalf("Ptp = %v want 8", got)
	}
}
