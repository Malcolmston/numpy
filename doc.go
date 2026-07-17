// Package numpy is a small, dependency-free n-dimensional array library for Go
// modeled on the core of Python's NumPy. It provides a single dense array type,
// NDArray, holding float64 values, together with creation helpers, shape
// manipulation, broadcasting element-wise math, axis reductions, basic linear
// algebra, and boolean masking.
//
// # The NDArray model
//
// An NDArray stores its elements in one flat []float64 slice. The logical view
// of that buffer is defined by three pieces of metadata:
//
//   - shape: the length of the array along each axis.
//   - strides: how many flat elements to advance to move one step along each
//     axis. Strides are expressed in elements, not bytes.
//   - ndim and size: cached number of axes and total element count.
//
// Newly constructed arrays are contiguous in row-major (C) order. Some
// operations return views that share the underlying buffer without copying:
// Transpose permutes axes by permuting strides, and Slice selects a sub-range
// by adjusting the offset and shape. Because views alias their parent's data,
// writing through a view is visible in the parent. Operations that must produce
// independent data (arithmetic, reductions, Reshape, Ravel, Copy) always return
// a fresh contiguous array.
//
// # Creation
//
// FromSlice, FromData and FromNested build arrays from existing values. Zeros,
// Ones, Full, Arange, Linspace, Eye and Identity generate arrays. Reshape
// re-views the same data under a new shape (with one optional inferred -1 axis).
//
// # Indexing and manipulation
//
// At and Set read and write single elements by multi-index (negative indices
// count from the end). Transpose, Ravel/Flatten, Slice, Concatenate and Stack
// reshape and combine arrays.
//
// # Broadcasting
//
// Binary element-wise operations broadcast their operands following NumPy's
// rules. Shapes are aligned from the trailing axis; two dimensions are
// compatible when they are equal or one of them is 1, and a size-1 dimension is
// virtually repeated (implemented with a zero stride, so no data is copied
// during the broadcast itself). For example a (3, 1) array and a (1, 4) array
// broadcast to (3, 4). Incompatible shapes panic.
//
// # Element-wise math and reductions
//
// Add, Sub, Mul, Div, Pow and the comparison operators broadcast two arrays;
// the *Scalar variants combine an array with a constant. Neg, Abs, Sqrt, Exp,
// Log, Sin and Cos map a function over every element. Sum, Mean, Max, Min, Std,
// Var and Prod reduce the whole array to a scalar, while SumAxis, MeanAxis,
// MaxAxis, MinAxis, StdAxis and ProdAxis reduce along one axis with an optional
// keepdims flag.
//
// # Linear algebra and masking
//
// Dot computes a 1-D dot product or a 2-D matrix product, and MatMul multiplies
// two matrices. The comparison operators return boolean masks encoded as 1.0 /
// 0.0 float64 arrays; MaskSelect gathers the elements where a mask is set and
// Where chooses element-wise between two arrays.
//
// # Errors
//
// The library is deterministic. All shape and dimension validation failures
// panic with a message prefixed by "numpy:". This keeps the arithmetic API free
// of error returns so operations can be chained; wrap calls with recover if you
// need to trap invalid usage.
package numpy
