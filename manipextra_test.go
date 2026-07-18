package numpy

import (
	"reflect"
	"testing"
)

func TestFlip(t *testing.T) {
	a := FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if got := a.Flip(0).Data(); !reflect.DeepEqual(got, []float64{4, 5, 6, 1, 2, 3}) {
		t.Fatalf("Flip(0) = %v", got)
	}
	if got := a.Flip(1).Data(); !reflect.DeepEqual(got, []float64{3, 2, 1, 6, 5, 4}) {
		t.Fatalf("Flip(1) = %v", got)
	}
	if got := a.Flip(-1).Data(); !reflect.DeepEqual(got, []float64{3, 2, 1, 6, 5, 4}) {
		t.Fatalf("Flip(-1) = %v", got)
	}
}

func TestRoll(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4})
	if got := a.Roll(2).Data(); !reflect.DeepEqual(got, []float64{3, 4, 0, 1, 2}) {
		t.Fatalf("Roll(2) = %v", got)
	}
	if got := a.Roll(-1).Data(); !reflect.DeepEqual(got, []float64{1, 2, 3, 4, 0}) {
		t.Fatalf("Roll(-1) = %v", got)
	}
	// Shape is preserved for multi-dimensional input.
	m := FromData([]float64{1, 2, 3, 4}, 2, 2)
	rolled := m.Roll(1)
	if !reflect.DeepEqual(rolled.Shape(), []int{2, 2}) {
		t.Fatalf("Roll shape = %v", rolled.Shape())
	}
	if !reflect.DeepEqual(rolled.Data(), []float64{4, 1, 2, 3}) {
		t.Fatalf("Roll 2D = %v", rolled.Data())
	}
}

func TestSqueeze(t *testing.T) {
	a := Zeros(1, 3, 1)
	got := a.Squeeze()
	if !reflect.DeepEqual(got.Shape(), []int{3}) {
		t.Fatalf("Squeeze shape = %v want [3]", got.Shape())
	}
	// All-ones shape collapses to a single element.
	b := Zeros(1, 1, 1).Squeeze()
	if !reflect.DeepEqual(b.Shape(), []int{1}) {
		t.Fatalf("Squeeze all-ones shape = %v want [1]", b.Shape())
	}
}

func TestExpandDims(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	if got := a.ExpandDims(0); !reflect.DeepEqual(got.Shape(), []int{1, 3}) {
		t.Fatalf("ExpandDims(0) shape = %v", got.Shape())
	}
	if got := a.ExpandDims(1); !reflect.DeepEqual(got.Shape(), []int{3, 1}) {
		t.Fatalf("ExpandDims(1) shape = %v", got.Shape())
	}
	if got := a.ExpandDims(-1); !reflect.DeepEqual(got.Shape(), []int{3, 1}) {
		t.Fatalf("ExpandDims(-1) shape = %v", got.Shape())
	}
	// Data is preserved.
	if got := a.ExpandDims(0).Data(); !reflect.DeepEqual(got, []float64{1, 2, 3}) {
		t.Fatalf("ExpandDims data = %v", got)
	}
}
