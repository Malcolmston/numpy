package numpy

// Boolean masks are represented as ordinary float64 arrays holding 1 for true
// and 0 for false, so they compose with the usual arithmetic and reductions.

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// Greater returns a mask that is 1 where a > b, with broadcasting.
func (a *NDArray) Greater(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return boolToFloat(x > y) })
}

// GreaterEqual returns a mask that is 1 where a >= b, with broadcasting.
func (a *NDArray) GreaterEqual(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return boolToFloat(x >= y) })
}

// Less returns a mask that is 1 where a < b, with broadcasting.
func (a *NDArray) Less(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return boolToFloat(x < y) })
}

// LessEqual returns a mask that is 1 where a <= b, with broadcasting.
func (a *NDArray) LessEqual(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return boolToFloat(x <= y) })
}

// EqualMask returns a mask that is 1 where a == b, with broadcasting.
func (a *NDArray) EqualMask(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return boolToFloat(x == y) })
}

// NotEqualMask returns a mask that is 1 where a != b, with broadcasting.
func (a *NDArray) NotEqualMask(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return boolToFloat(x != y) })
}

// GreaterScalar returns a mask that is 1 where a > s.
func (a *NDArray) GreaterScalar(s float64) *NDArray {
	return unaryOp(a, func(x float64) float64 { return boolToFloat(x > s) })
}

// LessScalar returns a mask that is 1 where a < s.
func (a *NDArray) LessScalar(s float64) *NDArray {
	return unaryOp(a, func(x float64) float64 { return boolToFloat(x < s) })
}

// EqualScalar returns a mask that is 1 where a == s.
func (a *NDArray) EqualScalar(s float64) *NDArray {
	return unaryOp(a, func(x float64) float64 { return boolToFloat(x == s) })
}

// MaskSelect returns a 1-D array of the elements of a where mask is non-zero,
// in row-major order. The mask must be broadcast-compatible with a; it is
// broadcast to a's shape before selection.
func (a *NDArray) MaskSelect(mask *NDArray) *NDArray {
	m := mask
	if !sameShape(a.shape, mask.shape) {
		m = mask.broadcastView(a.shape)
	}
	ad := a.Data()
	md := m.Data()
	out := make([]float64, 0, len(ad))
	for i, v := range md {
		if v != 0 {
			out = append(out, ad[i])
		}
	}
	return newArray(out, []int{len(out)})
}

// Where returns an array selecting from x where mask is non-zero and from y
// otherwise. All three are broadcast together.
func Where(mask, x, y *NDArray) *NDArray {
	target := broadcastShape(broadcastShape(mask.shape, x.shape), y.shape)
	mv := mask.broadcastView(target)
	xv := x.broadcastView(target)
	yv := y.broadcastView(target)
	out := Zeros(target...)
	md, xd, yd := mv.Data(), xv.Data(), yv.Data()
	for i := range md {
		if md[i] != 0 {
			out.data[i] = xd[i]
		} else {
			out.data[i] = yd[i]
		}
	}
	return out
}

// Any reports whether any element is non-zero.
func (a *NDArray) Any() bool {
	res := false
	a.forEach(func(off int, _ []int) {
		if a.data[off] != 0 {
			res = true
		}
	})
	return res
}

// All reports whether every element is non-zero.
func (a *NDArray) All() bool {
	res := true
	a.forEach(func(off int, _ []int) {
		if a.data[off] == 0 {
			res = false
		}
	})
	return res
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
