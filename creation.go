package numpy

import (
	"fmt"
	"math"
)

// FromSlice creates a 1-D array from a copy of the given values.
func FromSlice(values []float64) *NDArray {
	data := make([]float64, len(values))
	copy(data, values)
	return newArray(data, []int{len(values)})
}

// FromData creates an array from a copy of data laid out in row-major order
// for the given shape. It panics if len(data) does not match the shape size.
func FromData(data []float64, shape ...int) *NDArray {
	d := make([]float64, len(data))
	copy(d, data)
	return newArray(d, shape)
}

// FromNested creates an array from an arbitrarily nested slice of float64
// (for example []float64, [][]float64, [][][]float64). The nested value must
// be rectangular; a ragged structure panics.
func FromNested(nested any) *NDArray {
	shape := []int{}
	if err := inferShape(nested, &shape, 0); err != nil {
		panic(err.Error())
	}
	data := make([]float64, 0, sizeOf(shape))
	flatten(nested, &data)
	return newArray(data, shape)
}

func inferShape(v any, shape *[]int, depth int) error {
	switch t := v.(type) {
	case float64:
		return nil
	case []float64:
		if depth == len(*shape) {
			*shape = append(*shape, len(t))
		} else if (*shape)[depth] != len(t) {
			return fmt.Errorf("numpy: ragged nested slice at depth %d", depth)
		}
		return nil
	case []any:
		if depth == len(*shape) {
			*shape = append(*shape, len(t))
		} else if (*shape)[depth] != len(t) {
			return fmt.Errorf("numpy: ragged nested slice at depth %d", depth)
		}
		for _, e := range t {
			if err := inferShape(e, shape, depth+1); err != nil {
				return err
			}
		}
		return nil
	default:
		// Handle typed nested slices such as [][]float64 via reflection-free
		// recursion on []any is not possible, so support the common concrete
		// types explicitly below.
		return inferShapeConcrete(v, shape, depth)
	}
}

func inferShapeConcrete(v any, shape *[]int, depth int) error {
	switch t := v.(type) {
	case [][]float64:
		if depth == len(*shape) {
			*shape = append(*shape, len(t))
		} else if (*shape)[depth] != len(t) {
			return fmt.Errorf("numpy: ragged nested slice at depth %d", depth)
		}
		for _, e := range t {
			if err := inferShape(e, shape, depth+1); err != nil {
				return err
			}
		}
		return nil
	case [][][]float64:
		if depth == len(*shape) {
			*shape = append(*shape, len(t))
		} else if (*shape)[depth] != len(t) {
			return fmt.Errorf("numpy: ragged nested slice at depth %d", depth)
		}
		for _, e := range t {
			if err := inferShape(e, shape, depth+1); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("numpy: unsupported nested type %T", v)
	}
}

func flatten(v any, out *[]float64) {
	switch t := v.(type) {
	case float64:
		*out = append(*out, t)
	case []float64:
		*out = append(*out, t...)
	case []any:
		for _, e := range t {
			flatten(e, out)
		}
	case [][]float64:
		for _, e := range t {
			flatten(e, out)
		}
	case [][][]float64:
		for _, e := range t {
			flatten(e, out)
		}
	}
}

// Zeros returns a new array of the given shape filled with 0.
func Zeros(shape ...int) *NDArray {
	return newArray(make([]float64, sizeOf(shape)), shape)
}

// Ones returns a new array of the given shape filled with 1.
func Ones(shape ...int) *NDArray {
	return Full(1, shape...)
}

// Full returns a new array of the given shape filled with value.
func Full(value float64, shape ...int) *NDArray {
	data := make([]float64, sizeOf(shape))
	for i := range data {
		data[i] = value
	}
	return newArray(data, shape)
}

// ZerosLike returns a zero-filled array with the same shape as a.
func ZerosLike(a *NDArray) *NDArray { return Zeros(a.shape...) }

// OnesLike returns a one-filled array with the same shape as a.
func OnesLike(a *NDArray) *NDArray { return Ones(a.shape...) }

// Arange returns a 1-D array of evenly spaced values in the half-open interval
// [start, stop) advancing by step. It panics if step is zero.
func Arange(start, stop, step float64) *NDArray {
	if step == 0 {
		panic("numpy: Arange step must be non-zero")
	}
	var n int
	if step > 0 {
		if stop > start {
			n = int(math.Ceil((stop - start) / step))
		}
	} else {
		if stop < start {
			n = int(math.Ceil((stop - start) / step))
		}
	}
	if n < 0 {
		n = 0
	}
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = start + float64(i)*step
	}
	return newArray(data, []int{n})
}

// Linspace returns a 1-D array of num evenly spaced values. When endpoint is
// true, stop is the final value; otherwise stop is excluded. It panics if
// num is negative.
func Linspace(start, stop float64, num int, endpoint bool) *NDArray {
	if num < 0 {
		panic("numpy: Linspace num must be non-negative")
	}
	data := make([]float64, num)
	if num == 1 {
		data[0] = start
		return newArray(data, []int{1})
	}
	div := num
	if endpoint {
		div = num - 1
	}
	var step float64
	if div > 0 {
		step = (stop - start) / float64(div)
	}
	for i := 0; i < num; i++ {
		data[i] = start + float64(i)*step
	}
	if endpoint && num > 0 {
		data[num-1] = stop
	}
	return newArray(data, []int{num})
}

// Eye returns an n-by-m matrix with ones on the k-th diagonal and zeros
// elsewhere. A positive k is above the main diagonal, negative below.
func Eye(n, m, k int) *NDArray {
	if n < 0 || m < 0 {
		panic("numpy: Eye dimensions must be non-negative")
	}
	a := Zeros(n, m)
	for i := 0; i < n; i++ {
		j := i + k
		if j >= 0 && j < m {
			a.data[i*a.strides[0]+j*a.strides[1]] = 1
		}
	}
	return a
}

// Identity returns the n-by-n identity matrix.
func Identity(n int) *NDArray { return Eye(n, n, 0) }
