package numpy

import "math"

// Add returns a + b with broadcasting.
func (a *NDArray) Add(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return x + y })
}

// Sub returns a - b with broadcasting.
func (a *NDArray) Sub(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return x - y })
}

// Mul returns the element-wise product a * b with broadcasting.
func (a *NDArray) Mul(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return x * y })
}

// Div returns the element-wise quotient a / b with broadcasting.
func (a *NDArray) Div(b *NDArray) *NDArray {
	return binaryOp(a, b, func(x, y float64) float64 { return x / y })
}

// Pow returns a raised element-wise to the power b, with broadcasting.
func (a *NDArray) Pow(b *NDArray) *NDArray {
	return binaryOp(a, b, math.Pow)
}

// AddScalar returns a + s.
func (a *NDArray) AddScalar(s float64) *NDArray {
	return unaryOp(a, func(x float64) float64 { return x + s })
}

// SubScalar returns a - s.
func (a *NDArray) SubScalar(s float64) *NDArray {
	return unaryOp(a, func(x float64) float64 { return x - s })
}

// MulScalar returns a * s.
func (a *NDArray) MulScalar(s float64) *NDArray {
	return unaryOp(a, func(x float64) float64 { return x * s })
}

// DivScalar returns a / s.
func (a *NDArray) DivScalar(s float64) *NDArray {
	return unaryOp(a, func(x float64) float64 { return x / s })
}

// PowScalar returns a raised element-wise to the power s.
func (a *NDArray) PowScalar(s float64) *NDArray {
	return unaryOp(a, func(x float64) float64 { return math.Pow(x, s) })
}

// Neg returns the element-wise negation of a.
func (a *NDArray) Neg() *NDArray {
	return unaryOp(a, func(x float64) float64 { return -x })
}

// Abs returns the element-wise absolute value of a.
func (a *NDArray) Abs() *NDArray { return unaryOp(a, math.Abs) }

// Sqrt returns the element-wise square root of a.
func (a *NDArray) Sqrt() *NDArray { return unaryOp(a, math.Sqrt) }

// Exp returns the element-wise base-e exponential of a.
func (a *NDArray) Exp() *NDArray { return unaryOp(a, math.Exp) }

// Log returns the element-wise natural logarithm of a.
func (a *NDArray) Log() *NDArray { return unaryOp(a, math.Log) }

// Sin returns the element-wise sine of a.
func (a *NDArray) Sin() *NDArray { return unaryOp(a, math.Sin) }

// Cos returns the element-wise cosine of a.
func (a *NDArray) Cos() *NDArray { return unaryOp(a, math.Cos) }

// Maximum returns the element-wise maximum of a and b with broadcasting.
func (a *NDArray) Maximum(b *NDArray) *NDArray {
	return binaryOp(a, b, math.Max)
}

// Minimum returns the element-wise minimum of a and b with broadcasting.
func (a *NDArray) Minimum(b *NDArray) *NDArray {
	return binaryOp(a, b, math.Min)
}
