package numpy

import (
	"math"
	"testing"
)

func floatEq(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

func slicesEqInt(a, b []int) bool {
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

func slicesEqF(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !floatEq(a[i], b[i]) {
			return false
		}
	}
	return true
}

func mustPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

func TestCreation(t *testing.T) {
	z := Zeros(2, 3)
	if z.Ndim() != 2 || z.Size() != 6 || !slicesEqInt(z.Shape(), []int{2, 3}) {
		t.Fatalf("Zeros metadata wrong: %v", z)
	}
	if !slicesEqInt(z.Strides(), []int{3, 1}) {
		t.Fatalf("strides wrong: %v", z.Strides())
	}
	o := Ones(2, 2)
	if !slicesEqF(o.Data(), []float64{1, 1, 1, 1}) {
		t.Fatalf("Ones wrong: %v", o.Data())
	}
	f := Full(7, 3)
	if !slicesEqF(f.Data(), []float64{7, 7, 7}) {
		t.Fatalf("Full wrong")
	}
	if !slicesEqF(ZerosLike(o).Data(), []float64{0, 0, 0, 0}) {
		t.Fatal("ZerosLike wrong")
	}
	if !slicesEqF(OnesLike(z).Data(), []float64{1, 1, 1, 1, 1, 1}) {
		t.Fatal("OnesLike wrong")
	}
	fs := FromSlice([]float64{1, 2, 3})
	if !slicesEqF(fs.Data(), []float64{1, 2, 3}) {
		t.Fatal("FromSlice wrong")
	}
	fd := FromData([]float64{1, 2, 3, 4}, 2, 2)
	if !slicesEqInt(fd.Shape(), []int{2, 2}) {
		t.Fatal("FromData wrong")
	}
	mustPanic(t, "FromData mismatch", func() { FromData([]float64{1, 2, 3}, 2, 2) })
	mustPanic(t, "negative dim", func() { Zeros(-1, 2) })
}

func TestFromNested(t *testing.T) {
	a := FromNested([][]float64{{1, 2, 3}, {4, 5, 6}})
	if !slicesEqInt(a.Shape(), []int{2, 3}) || !slicesEqF(a.Data(), []float64{1, 2, 3, 4, 5, 6}) {
		t.Fatalf("FromNested 2D wrong: %v", a)
	}
	b := FromNested([][][]float64{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}})
	if !slicesEqInt(b.Shape(), []int{2, 2, 2}) {
		t.Fatalf("FromNested 3D shape wrong: %v", b.Shape())
	}
	c := FromNested([]float64{1, 2, 3})
	if !slicesEqInt(c.Shape(), []int{3}) {
		t.Fatal("FromNested 1D wrong")
	}
	d := FromNested([]any{[]float64{1, 2}, []float64{3, 4}})
	if !slicesEqInt(d.Shape(), []int{2, 2}) {
		t.Fatalf("FromNested any wrong: %v", d.Shape())
	}
	mustPanic(t, "ragged", func() { FromNested([][]float64{{1, 2}, {3}}) })
	mustPanic(t, "unsupported", func() { FromNested([]int{1, 2}) })
}

func TestArangeLinspace(t *testing.T) {
	if !slicesEqF(Arange(0, 5, 1).Data(), []float64{0, 1, 2, 3, 4}) {
		t.Fatal("Arange asc wrong")
	}
	if !slicesEqF(Arange(5, 0, -2).Data(), []float64{5, 3, 1}) {
		t.Fatalf("Arange desc wrong: %v", Arange(5, 0, -2).Data())
	}
	if Arange(5, 0, 1).Size() != 0 {
		t.Fatal("Arange empty wrong")
	}
	mustPanic(t, "arange step 0", func() { Arange(0, 1, 0) })

	if !slicesEqF(Linspace(0, 1, 5, true).Data(), []float64{0, 0.25, 0.5, 0.75, 1}) {
		t.Fatalf("Linspace endpoint wrong: %v", Linspace(0, 1, 5, true).Data())
	}
	if !slicesEqF(Linspace(0, 1, 5, false).Data(), []float64{0, 0.2, 0.4, 0.6, 0.8}) {
		t.Fatalf("Linspace no endpoint wrong: %v", Linspace(0, 1, 5, false).Data())
	}
	if !slicesEqF(Linspace(3, 9, 1, true).Data(), []float64{3}) {
		t.Fatal("Linspace single wrong")
	}
	if Linspace(0, 1, 0, true).Size() != 0 {
		t.Fatal("Linspace zero wrong")
	}
	mustPanic(t, "linspace neg", func() { Linspace(0, 1, -1, true) })
}

func TestEyeIdentity(t *testing.T) {
	if !slicesEqF(Identity(3).Data(), []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}) {
		t.Fatal("Identity wrong")
	}
	if !slicesEqF(Eye(2, 3, 1).Data(), []float64{0, 1, 0, 0, 0, 1}) {
		t.Fatalf("Eye k=1 wrong: %v", Eye(2, 3, 1).Data())
	}
	if !slicesEqF(Eye(3, 3, -1).Data(), []float64{0, 0, 0, 1, 0, 0, 0, 1, 0}) {
		t.Fatalf("Eye k=-1 wrong: %v", Eye(3, 3, -1).Data())
	}
	mustPanic(t, "eye neg", func() { Eye(-1, 2, 0) })
}

func TestAtSet(t *testing.T) {
	a := Arange(0, 6, 1).Reshape(2, 3)
	if a.At(1, 2) != 5 {
		t.Fatalf("At wrong: %v", a.At(1, 2))
	}
	if a.At(-1, -1) != 5 {
		t.Fatal("At negative wrong")
	}
	a.Set(99, 0, 0)
	if a.At(0, 0) != 99 {
		t.Fatal("Set wrong")
	}
	mustPanic(t, "index oob", func() { a.At(5, 5) })
	mustPanic(t, "wrong ndim", func() { a.At(1) })
}

func TestReshapeRavel(t *testing.T) {
	a := Arange(0, 12, 1)
	r := a.Reshape(3, 4)
	if !slicesEqInt(r.Shape(), []int{3, 4}) {
		t.Fatal("Reshape wrong")
	}
	inferred := a.Reshape(2, -1)
	if !slicesEqInt(inferred.Shape(), []int{2, 6}) {
		t.Fatalf("Reshape infer wrong: %v", inferred.Shape())
	}
	fl := r.Flatten()
	if fl.Ndim() != 1 || fl.Size() != 12 {
		t.Fatal("Flatten wrong")
	}
	if !slicesEqF(r.Ravel().Data(), a.Data()) {
		t.Fatal("Ravel wrong")
	}
	mustPanic(t, "reshape mismatch", func() { a.Reshape(5, 5) })
	mustPanic(t, "two infer", func() { a.Reshape(-1, -1) })
	mustPanic(t, "neg dim", func() { a.Reshape(-2, 6) })
}

func TestTranspose(t *testing.T) {
	a := Arange(0, 6, 1).Reshape(2, 3)
	tr := a.T()
	if !slicesEqInt(tr.Shape(), []int{3, 2}) {
		t.Fatalf("T shape wrong: %v", tr.Shape())
	}
	if !slicesEqF(tr.Data(), []float64{0, 3, 1, 4, 2, 5}) {
		t.Fatalf("T data wrong: %v", tr.Data())
	}
	b := Arange(0, 24, 1).Reshape(2, 3, 4)
	p := b.Transpose(2, 0, 1)
	if !slicesEqInt(p.Shape(), []int{4, 2, 3}) {
		t.Fatalf("Transpose perm shape wrong: %v", p.Shape())
	}
	if p.At(3, 1, 2) != b.At(1, 2, 3) {
		t.Fatal("Transpose value mapping wrong")
	}
	mustPanic(t, "bad axes len", func() { a.Transpose(0) })
	mustPanic(t, "dup axes", func() { a.Transpose(0, 0) })
}

func TestSlice(t *testing.T) {
	a := Arange(0, 12, 1).Reshape(3, 4)
	s := a.Slice(R(1, 3), R(1, 3))
	if !slicesEqInt(s.Shape(), []int{2, 2}) {
		t.Fatalf("Slice shape wrong: %v", s.Shape())
	}
	if !slicesEqF(s.Data(), []float64{5, 6, 9, 10}) {
		t.Fatalf("Slice data wrong: %v", s.Data())
	}
	full := a.Slice(R(0, 0), R(0, 0))
	if !slicesEqF(full.Data(), a.Data()) {
		t.Fatal("Slice full wrong")
	}
	neg := a.Slice(R(-2, -1), R(0, 0))
	if !slicesEqF(neg.Data(), []float64{4, 5, 6, 7}) {
		t.Fatalf("Slice neg wrong: %v", neg.Data())
	}
	mustPanic(t, "slice ndim", func() { a.Slice(R(0, 1)) })
}

func TestConcatenateStack(t *testing.T) {
	a := Arange(0, 4, 1).Reshape(2, 2)
	b := Arange(4, 8, 1).Reshape(2, 2)
	c0 := Concatenate(0, a, b)
	if !slicesEqInt(c0.Shape(), []int{4, 2}) || !slicesEqF(c0.Data(), []float64{0, 1, 2, 3, 4, 5, 6, 7}) {
		t.Fatalf("Concatenate axis0 wrong: %v", c0)
	}
	c1 := Concatenate(1, a, b)
	if !slicesEqInt(c1.Shape(), []int{2, 4}) || !slicesEqF(c1.Data(), []float64{0, 1, 4, 5, 2, 3, 6, 7}) {
		t.Fatalf("Concatenate axis1 wrong: %v", c1.Data())
	}
	cneg := Concatenate(-1, a, b)
	if !slicesEqInt(cneg.Shape(), []int{2, 4}) {
		t.Fatal("Concatenate neg axis wrong")
	}
	s0 := Stack(0, a, b)
	if !slicesEqInt(s0.Shape(), []int{2, 2, 2}) {
		t.Fatalf("Stack axis0 shape wrong: %v", s0.Shape())
	}
	s2 := Stack(2, a, b)
	if !slicesEqInt(s2.Shape(), []int{2, 2, 2}) || s2.At(0, 0, 1) != 4 {
		t.Fatalf("Stack axis2 wrong: %v", s2)
	}
	mustPanic(t, "concat empty", func() { Concatenate(0) })
	mustPanic(t, "concat mismatch", func() { Concatenate(0, a, Zeros(2, 3)) })
	mustPanic(t, "concat axis oob", func() { Concatenate(5, a, b) })
	mustPanic(t, "stack empty", func() { Stack(0) })
	mustPanic(t, "stack mismatch", func() { Stack(0, a, Zeros(3, 3)) })
	mustPanic(t, "stack axis oob", func() { Stack(9, a, b) })
}

func TestBroadcasting(t *testing.T) {
	a := Arange(0, 6, 1).Reshape(2, 3)
	row := FromSlice([]float64{10, 20, 30})
	if !slicesEqF(a.Add(row).Data(), []float64{10, 21, 32, 13, 24, 35}) {
		t.Fatalf("row broadcast wrong: %v", a.Add(row).Data())
	}
	col := FromData([]float64{100, 200}, 2, 1)
	if !slicesEqF(a.Add(col).Data(), []float64{100, 101, 102, 203, 204, 205}) {
		t.Fatalf("col broadcast wrong: %v", a.Add(col).Data())
	}
	// (3,1) + (1,4) -> (3,4)
	x := FromData([]float64{0, 1, 2}, 3, 1)
	y := FromData([]float64{0, 10, 20, 30}, 1, 4)
	z := x.Add(y)
	if !slicesEqInt(z.Shape(), []int{3, 4}) {
		t.Fatalf("outer broadcast shape wrong: %v", z.Shape())
	}
	if !slicesEqF(z.Data(), []float64{0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32}) {
		t.Fatalf("outer broadcast data wrong: %v", z.Data())
	}
	bt := row.BroadcastTo(2, 3)
	if !slicesEqF(bt.Data(), []float64{10, 20, 30, 10, 20, 30}) {
		t.Fatalf("BroadcastTo wrong: %v", bt.Data())
	}
	mustPanic(t, "incompatible", func() { a.Add(Zeros(4, 4)) })
}

func TestElementwiseMath(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4})
	b := FromSlice([]float64{4, 3, 2, 1})
	if !slicesEqF(a.Sub(b).Data(), []float64{-3, -1, 1, 3}) {
		t.Fatal("Sub wrong")
	}
	if !slicesEqF(a.Mul(b).Data(), []float64{4, 6, 6, 4}) {
		t.Fatal("Mul wrong")
	}
	if !slicesEqF(a.Div(b).Data(), []float64{0.25, 2.0 / 3, 1.5, 4}) {
		t.Fatal("Div wrong")
	}
	if !slicesEqF(a.Pow(FromSlice([]float64{2, 2, 2, 2})).Data(), []float64{1, 4, 9, 16}) {
		t.Fatal("Pow wrong")
	}
	if !slicesEqF(a.AddScalar(10).Data(), []float64{11, 12, 13, 14}) {
		t.Fatal("AddScalar wrong")
	}
	if !slicesEqF(a.SubScalar(1).Data(), []float64{0, 1, 2, 3}) {
		t.Fatal("SubScalar wrong")
	}
	if !slicesEqF(a.MulScalar(2).Data(), []float64{2, 4, 6, 8}) {
		t.Fatal("MulScalar wrong")
	}
	if !slicesEqF(a.DivScalar(2).Data(), []float64{0.5, 1, 1.5, 2}) {
		t.Fatal("DivScalar wrong")
	}
	if !slicesEqF(a.PowScalar(2).Data(), []float64{1, 4, 9, 16}) {
		t.Fatal("PowScalar wrong")
	}
	if !slicesEqF(a.Neg().Data(), []float64{-1, -2, -3, -4}) {
		t.Fatal("Neg wrong")
	}
	if !slicesEqF(a.Neg().Abs().Data(), []float64{1, 2, 3, 4}) {
		t.Fatal("Abs wrong")
	}
	if !slicesEqF(FromSlice([]float64{1, 4, 9}).Sqrt().Data(), []float64{1, 2, 3}) {
		t.Fatal("Sqrt wrong")
	}
	if !floatEq(FromSlice([]float64{0}).Exp().Data()[0], 1) {
		t.Fatal("Exp wrong")
	}
	if !floatEq(FromSlice([]float64{1}).Log().Data()[0], 0) {
		t.Fatal("Log wrong")
	}
	if !floatEq(FromSlice([]float64{0}).Sin().Data()[0], 0) {
		t.Fatal("Sin wrong")
	}
	if !floatEq(FromSlice([]float64{0}).Cos().Data()[0], 1) {
		t.Fatal("Cos wrong")
	}
	if !slicesEqF(a.Maximum(b).Data(), []float64{4, 3, 3, 4}) {
		t.Fatal("Maximum wrong")
	}
	if !slicesEqF(a.Minimum(b).Data(), []float64{1, 2, 2, 1}) {
		t.Fatal("Minimum wrong")
	}
}

func TestReductionsWhole(t *testing.T) {
	a := Arange(1, 5, 1) // 1,2,3,4
	if a.Sum() != 10 {
		t.Fatal("Sum wrong")
	}
	if a.Prod() != 24 {
		t.Fatal("Prod wrong")
	}
	if a.Mean() != 2.5 {
		t.Fatal("Mean wrong")
	}
	if a.Max() != 4 || a.Min() != 1 {
		t.Fatal("Max/Min wrong")
	}
	if !floatEq(a.Var(), 1.25) {
		t.Fatalf("Var wrong: %v", a.Var())
	}
	if !floatEq(a.Std(), math.Sqrt(1.25)) {
		t.Fatal("Std wrong")
	}
	mustPanic(t, "mean empty", func() { Zeros(0).Mean() })
	mustPanic(t, "max empty", func() { Zeros(0).Max() })
	mustPanic(t, "min empty", func() { Zeros(0).Min() })
}

func TestReductionsAxis(t *testing.T) {
	a := Arange(0, 6, 1).Reshape(2, 3) // [[0,1,2],[3,4,5]]
	if !slicesEqF(a.SumAxis(0, false).Data(), []float64{3, 5, 7}) {
		t.Fatalf("SumAxis0 wrong: %v", a.SumAxis(0, false).Data())
	}
	if !slicesEqF(a.SumAxis(1, false).Data(), []float64{3, 12}) {
		t.Fatalf("SumAxis1 wrong: %v", a.SumAxis(1, false).Data())
	}
	kd := a.SumAxis(1, true)
	if !slicesEqInt(kd.Shape(), []int{2, 1}) {
		t.Fatalf("keepdims shape wrong: %v", kd.Shape())
	}
	if !slicesEqF(a.MeanAxis(0, false).Data(), []float64{1.5, 2.5, 3.5}) {
		t.Fatal("MeanAxis wrong")
	}
	if !slicesEqF(a.MaxAxis(1, false).Data(), []float64{2, 5}) {
		t.Fatal("MaxAxis wrong")
	}
	if !slicesEqF(a.MinAxis(0, false).Data(), []float64{0, 1, 2}) {
		t.Fatal("MinAxis wrong")
	}
	if !slicesEqF(a.ProdAxis(1, false).Data(), []float64{0, 60}) {
		t.Fatalf("ProdAxis wrong: %v", a.ProdAxis(1, false).Data())
	}
	// StdAxis along axis 1: rows [0,1,2] and [3,4,5] each have std sqrt(2/3).
	want := math.Sqrt(2.0 / 3.0)
	sd := a.StdAxis(1, false)
	if !slicesEqF(sd.Data(), []float64{want, want}) {
		t.Fatalf("StdAxis wrong: %v", sd.Data())
	}
	// Negative axis.
	if !slicesEqF(a.SumAxis(-1, false).Data(), []float64{3, 12}) {
		t.Fatal("SumAxis neg wrong")
	}
	// 3-D reduction.
	b := Arange(0, 24, 1).Reshape(2, 3, 4)
	r := b.SumAxis(1, false)
	if !slicesEqInt(r.Shape(), []int{2, 4}) {
		t.Fatalf("3D SumAxis shape wrong: %v", r.Shape())
	}
	if !slicesEqF(r.Data(), []float64{12, 15, 18, 21, 48, 51, 54, 57}) {
		t.Fatalf("3D SumAxis data wrong: %v", r.Data())
	}
	mustPanic(t, "axis oob", func() { a.SumAxis(5, false) })
}

func TestLinalg(t *testing.T) {
	a := FromData([]float64{1, 2, 3, 4}, 2, 2)
	if !slicesEqF(a.MatMul(a).Data(), []float64{7, 10, 15, 22}) {
		t.Fatalf("MatMul wrong: %v", a.MatMul(a).Data())
	}
	b := FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c := FromData([]float64{7, 8, 9, 10, 11, 12}, 3, 2)
	m := b.MatMul(c)
	if !slicesEqInt(m.Shape(), []int{2, 2}) {
		t.Fatalf("MatMul rect shape wrong: %v", m.Shape())
	}
	if !slicesEqF(m.Data(), []float64{58, 64, 139, 154}) {
		t.Fatalf("MatMul rect data wrong: %v", m.Data())
	}
	// MatMul with a transposed (non-contiguous) operand.
	d := b.MatMul(b.T())
	if !slicesEqF(d.Data(), []float64{14, 32, 32, 77}) {
		t.Fatalf("MatMul with T wrong: %v", d.Data())
	}
	dot := FromSlice([]float64{1, 2, 3}).Dot(FromSlice([]float64{4, 5, 6}))
	if !slicesEqF(dot.Data(), []float64{32}) {
		t.Fatalf("Dot 1D wrong: %v", dot.Data())
	}
	if !slicesEqF(a.Dot(a).Data(), []float64{7, 10, 15, 22}) {
		t.Fatal("Dot 2D wrong")
	}
	mustPanic(t, "matmul ndim", func() { FromSlice([]float64{1, 2}).MatMul(a) })
	mustPanic(t, "matmul inner", func() { a.MatMul(FromData([]float64{1, 2, 3}, 3, 1)) })
	mustPanic(t, "dot len", func() { FromSlice([]float64{1, 2}).Dot(FromSlice([]float64{1})) })
	mustPanic(t, "dot ndim", func() { FromSlice([]float64{1}).Dot(a) })
}

func TestCompareMask(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4})
	b := FromSlice([]float64{4, 3, 2, 1})
	if !slicesEqF(a.Greater(b).Data(), []float64{0, 0, 1, 1}) {
		t.Fatal("Greater wrong")
	}
	if !slicesEqF(a.GreaterEqual(b).Data(), []float64{0, 0, 1, 1}) {
		t.Fatal("GreaterEqual wrong")
	}
	if !slicesEqF(a.Less(b).Data(), []float64{1, 1, 0, 0}) {
		t.Fatal("Less wrong")
	}
	if !slicesEqF(a.LessEqual(b).Data(), []float64{1, 1, 0, 0}) {
		t.Fatal("LessEqual wrong")
	}
	if !slicesEqF(a.EqualMask(b).Data(), []float64{0, 0, 0, 0}) {
		t.Fatal("EqualMask wrong")
	}
	if !slicesEqF(a.NotEqualMask(b).Data(), []float64{1, 1, 1, 1}) {
		t.Fatal("NotEqualMask wrong")
	}
	if !slicesEqF(a.GreaterScalar(2).Data(), []float64{0, 0, 1, 1}) {
		t.Fatal("GreaterScalar wrong")
	}
	if !slicesEqF(a.LessScalar(2).Data(), []float64{1, 0, 0, 0}) {
		t.Fatal("LessScalar wrong")
	}
	if !slicesEqF(a.EqualScalar(3).Data(), []float64{0, 0, 1, 0}) {
		t.Fatal("EqualScalar wrong")
	}
	sel := a.MaskSelect(a.GreaterScalar(2))
	if !slicesEqF(sel.Data(), []float64{3, 4}) {
		t.Fatalf("MaskSelect wrong: %v", sel.Data())
	}
	// Broadcast mask select.
	m := Arange(0, 6, 1).Reshape(2, 3)
	rowMask := FromSlice([]float64{1, 0, 1})
	bsel := m.MaskSelect(rowMask)
	if !slicesEqF(bsel.Data(), []float64{0, 2, 3, 5}) {
		t.Fatalf("Broadcast MaskSelect wrong: %v", bsel.Data())
	}
	w := Where(a.GreaterScalar(2), a, b)
	if !slicesEqF(w.Data(), []float64{4, 3, 3, 4}) {
		t.Fatalf("Where wrong: %v", w.Data())
	}
	if !a.GreaterScalar(0).All() {
		t.Fatal("All wrong")
	}
	if a.GreaterScalar(10).Any() {
		t.Fatal("Any wrong")
	}
	if !a.GreaterScalar(3).Any() {
		t.Fatal("Any true wrong")
	}
}

func TestUtilAndViews(t *testing.T) {
	a := Arange(0, 6, 1).Reshape(2, 3)
	cp := a.Copy()
	cp.Set(100, 0, 0)
	if a.At(0, 0) == 100 {
		t.Fatal("Copy should be independent")
	}
	if !a.Equal(Arange(0, 6, 1).Reshape(2, 3)) {
		t.Fatal("Equal wrong")
	}
	if a.Equal(Zeros(2, 3)) {
		t.Fatal("Equal false wrong")
	}
	if a.Equal(Zeros(6)) {
		t.Fatal("Equal ndim wrong")
	}
	if a.Equal(Zeros(2, 4)) {
		t.Fatal("Equal shape wrong")
	}
	if !a.AllClose(a.AddScalar(1e-12), 1e-9) {
		t.Fatal("AllClose wrong")
	}
	if a.AllClose(a.AddScalar(1), 1e-9) {
		t.Fatal("AllClose false wrong")
	}
	if a.AllClose(Zeros(6), 1e-9) || a.AllClose(Zeros(2, 4), 1e-9) {
		t.Fatal("AllClose shape guard wrong")
	}
	if a.String() == "" {
		t.Fatal("String empty")
	}
	// View aliasing: transpose shares data.
	tr := a.Transpose()
	tr.Set(42, 0, 0)
	if a.At(0, 0) != 42 {
		t.Fatal("Transpose view should alias data")
	}
	if !a.isContiguous() {
		t.Fatal("expected contiguous")
	}
	// A sliced sub-view is non-contiguous.
	sub := Arange(0, 12, 1).Reshape(3, 4).Slice(R(0, 2), R(1, 3))
	if sub.isContiguous() {
		t.Fatal("expected non-contiguous slice")
	}
	if Zeros(0).Sum() != 0 {
		t.Fatal("empty sum should be 0")
	}
}
