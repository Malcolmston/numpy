// Library content for the numpy documentation site. Mirrors the shape used by
// the malcolmston/go landing site's data.ts so the sibling sites stay in sync.
export interface Lib {
  id: string; name: string; icon: string; accent: string; pkg: string; node: string;
  repo: string; docs: string; tagline: string; blurb: string; tags: string[];
  features: string[]; node_code: string; go_code: string; integrate: string;
}

export const NODE_ACCENT = '#8cc84b';

export const NUMPY: Lib = {
  id:"numpy", name:"numpy", icon:'<i class="fa-solid fa-table-cells"></i>', accent:"#4dabcf",
  pkg:"github.com/malcolmston/numpy", node:"numpy/numpy",
  repo:"https://github.com/malcolmston/numpy", docs:"https://malcolmston.github.io/numpy/",
  tagline:"NumPy-style n-dimensional arrays in Go.",
  blurb:"A from-scratch, standard-library-only Go library modeled on the core of Python's NumPy. Everything is "+
    "built on a single dense NDArray of float64, backed by one flat slice with row-major shape and strides — "+
    "no cgo, no third-party dependencies. You get creation helpers, zero-copy views, NumPy-rule broadcasting "+
    "element-wise math, whole-array and per-axis reductions, basic linear algebra and boolean masking. "+
    "Transpose, Slice and Reshape return views that share the parent's buffer, while arithmetic and "+
    "reductions always return fresh contiguous arrays. All shape and dimension validation failures panic "+
    "with a message prefixed by \"numpy:\", keeping the arithmetic API free of error returns so operations "+
    "can be chained.",
  tags:["NDArray","float64","shape/strides","row-major","broadcasting","axis reductions","views","Reshape","MatMul","boolean masking","keepdims","stdlib-only"],
  features:[
    "<code>NDArray</code> core — a dense row-major array of <code>float64</code> with cached <code>Shape</code>, <code>Strides</code>, <code>Ndim</code> and <code>Size</code>",
    "Creation — <code>FromSlice</code>, <code>FromData</code>, <code>FromNested</code>, <code>Zeros</code>, <code>Ones</code>, <code>Full</code>, <code>Arange</code>, <code>Linspace</code>, <code>Eye</code>, <code>Identity</code>",
    "Zero-copy views — <code>Transpose</code>/<code>T</code> permute strides, <code>Slice</code> adjusts offset+shape, <code>Reshape</code> re-views the same buffer",
    "Broadcasting — NumPy's trailing-axis rules via <code>BroadcastTo</code>, with size-1 dimensions expanded through a zero stride (no copy)",
    "Element-wise math — <code>Add</code>, <code>Sub</code>, <code>Mul</code>, <code>Div</code>, <code>Pow</code> plus <code>*Scalar</code> variants, and <code>Neg</code>/<code>Abs</code>/<code>Sqrt</code>/<code>Exp</code>/<code>Log</code>/<code>Sin</code>/<code>Cos</code>",
    "Reductions — whole-array <code>Sum</code>, <code>Mean</code>, <code>Max</code>, <code>Min</code>, <code>Std</code>, <code>Var</code>, <code>Prod</code> and per-axis <code>SumAxis</code>/<code>MeanAxis</code>/<code>MaxAxis</code>/… with optional <code>keepdims</code>",
    "Linear algebra — <code>Dot</code> (1-D dot / 2-D matmul) and <code>MatMul</code> for matrix products",
    "Comparison &amp; masking — <code>Greater</code>, <code>Less</code>, <code>EqualMask</code> (and scalar variants), <code>MaskSelect</code>, <code>Where</code>, <code>Any</code>, <code>All</code>",
    "Indexing &amp; combining — <code>At</code>/<code>Set</code> by multi-index (negatives allowed), plus <code>Concatenate</code> and <code>Stack</code>",
    "Panic-based errors — every shape or dimension failure panics with a <code>numpy:</code> prefix, so the arithmetic API stays return-value clean and chainable",
    "Zero dependencies — pure Go standard library, nothing to audit but the toolchain"
  ],
  node_code:
`import numpy as np

a = np.arange(0, 6).reshape(2, 3)     # [[0 1 2] [3 4 5]]
b = np.array([10, 20, 30])            # broadcast (2,3) + (3,)
print(a + b)                          # [[10 21 32] [13 24 35]]
print(a.sum(axis=0))                  # [3 5 7]
print(a @ a.T)                        # [[ 5 14] [14 50]]
print(a[a > 2])                       # [3 4 5]`,
  go_code:
`import np "github.com/malcolmston/numpy"

a := np.Arange(0, 6, 1).Reshape(2, 3)      // [[0 1 2] [3 4 5]]
b := np.FromSlice([]float64{10, 20, 30})    // broadcast (2,3) + (3,)
fmt.Println(a.Add(b).Data())                // [10 21 32 13 24 35]
fmt.Println(a.SumAxis(0, false).Data())     // [3 5 7]
fmt.Println(a.MatMul(a.T()).Data())         // [5 14 14 50]
mask := a.GreaterScalar(2)
fmt.Println(a.MaskSelect(mask).Data())      // [3 4 5]`,
  integrate:
`<span class="tok-c">// Build a 2x3 grid and take a zero-copy transposed view — no data is</span>
<span class="tok-c">// copied, only the strides are permuted.</span>
a := np.Arange(0, 6, 1).Reshape(2, 3)
at := a.T() <span class="tok-c">// shape (3,2), shares a's buffer</span>

<span class="tok-c">// Broadcasting follows NumPy's trailing-axis rules: (3,1) + (1,2) -&gt; (3,2).</span>
col := np.FromData([]float64{1, 2, 3}, 3, 1)
row := np.FromData([]float64{10, 20}, 1, 2)
grid := col.Add(row)

<span class="tok-c">// Reduce along an axis with keepdims, then centre each column.</span>
means := a.MeanAxis(0, true)   <span class="tok-c">// shape (1,3)</span>
centered := a.Sub(means)       <span class="tok-c">// broadcast back to (2,3)</span>

<span class="tok-c">// Boolean masking: keep only the entries greater than the column mean.</span>
mask := centered.GreaterScalar(0)
kept := centered.MaskSelect(mask)

<span class="tok-c">// Matrix product of a (2x3) with its transpose (3x2) -&gt; (2x2) Gram matrix.</span>
gram := a.MatMul(at)`
};
