package numpy

import (
	"reflect"
	"testing"
)

func TestSort(t *testing.T) {
	a := FromData([]float64{3, 1, 2, 5, 4, 0}, 2, 3)
	got := a.Sort()
	if !reflect.DeepEqual(got.Shape(), []int{6}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{0, 1, 2, 3, 4, 5}
	if !reflect.DeepEqual(got.Data(), want) {
		t.Fatalf("Sort = %v want %v", got.Data(), want)
	}
}

func TestArgsort(t *testing.T) {
	a := FromSlice([]float64{3, 1, 2})
	got := a.Argsort().Data()
	want := []float64{1, 2, 0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Argsort = %v want %v", got, want)
	}
}

func TestArgmaxArgmin(t *testing.T) {
	a := FromData([]float64{1, 9, 3, 2}, 2, 2)
	if got := a.Argmax(); got != 1 {
		t.Fatalf("Argmax = %d want 1", got)
	}
	if got := a.Argmin(); got != 0 {
		t.Fatalf("Argmin = %d want 0", got)
	}
	// First occurrence on ties.
	b := FromSlice([]float64{5, 5, 1, 5})
	if got := b.Argmax(); got != 0 {
		t.Fatalf("Argmax ties = %d want 0", got)
	}
}

func TestUnique(t *testing.T) {
	a := FromSlice([]float64{3, 1, 2, 3, 1, 1, 4})
	got := a.Unique().Data()
	want := []float64{1, 2, 3, 4}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Unique = %v want %v", got, want)
	}
}

func TestSearchSorted(t *testing.T) {
	a := FromSlice([]float64{1, 2, 4, 4, 8})
	cases := []struct {
		v    float64
		want int
	}{
		{0, 0}, {1, 0}, {3, 2}, {4, 2}, {5, 4}, {9, 5},
	}
	for _, c := range cases {
		if got := a.SearchSorted(c.v); got != c.want {
			t.Fatalf("SearchSorted(%v) = %d want %d", c.v, got, c.want)
		}
	}
}

func TestSearchSortedPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for non-1-D array")
		}
	}()
	Zeros(2, 2).SearchSorted(1)
}

func BenchmarkSort(b *testing.B) {
	a := Arange(1000, 0, -1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Sort()
	}
}
