package numpy

import "math"

// Tan returns the element-wise tangent of a.
func (a *NDArray) Tan() *NDArray { return unaryOp(a, math.Tan) }

// Sinh returns the element-wise hyperbolic sine of a.
func (a *NDArray) Sinh() *NDArray { return unaryOp(a, math.Sinh) }

// Cosh returns the element-wise hyperbolic cosine of a.
func (a *NDArray) Cosh() *NDArray { return unaryOp(a, math.Cosh) }

// Tanh returns the element-wise hyperbolic tangent of a.
func (a *NDArray) Tanh() *NDArray { return unaryOp(a, math.Tanh) }

// Arcsin returns the element-wise inverse sine (in radians) of a.
func (a *NDArray) Arcsin() *NDArray { return unaryOp(a, math.Asin) }

// Arccos returns the element-wise inverse cosine (in radians) of a.
func (a *NDArray) Arccos() *NDArray { return unaryOp(a, math.Acos) }

// Arctan returns the element-wise inverse tangent (in radians) of a.
func (a *NDArray) Arctan() *NDArray { return unaryOp(a, math.Atan) }

// Arctan2 returns the element-wise arc tangent of a/b choosing the quadrant
// from the signs of both arguments, with broadcasting. It mirrors NumPy's
// arctan2 and math.Atan2 (a is the "y" argument, b is "x").
func (a *NDArray) Arctan2(b *NDArray) *NDArray {
	return binaryOp(a, b, math.Atan2)
}

// Hypot returns the element-wise sqrt(a*a + b*b) computed without undue
// overflow, with broadcasting.
func (a *NDArray) Hypot(b *NDArray) *NDArray {
	return binaryOp(a, b, math.Hypot)
}

// Mod returns the element-wise remainder of a/b with broadcasting, computed as
// math.Mod (the result has the sign of the dividend a).
func (a *NDArray) Mod(b *NDArray) *NDArray {
	return binaryOp(a, b, math.Mod)
}

// Floor returns the element-wise floor of a (greatest integer <= x).
func (a *NDArray) Floor() *NDArray { return unaryOp(a, math.Floor) }

// Ceil returns the element-wise ceiling of a (least integer >= x).
func (a *NDArray) Ceil() *NDArray { return unaryOp(a, math.Ceil) }

// Round returns the element-wise rounding of a to the nearest integer, with
// halves rounded to the nearest even value (banker's rounding), matching
// NumPy's default round behavior.
func (a *NDArray) Round() *NDArray { return unaryOp(a, math.RoundToEven) }

// Trunc returns the element-wise truncation of a toward zero.
func (a *NDArray) Trunc() *NDArray { return unaryOp(a, math.Trunc) }

// Sign returns the element-wise sign of a: -1 for negative values, 0 for zero,
// and 1 for positive values. A NaN input yields NaN.
func (a *NDArray) Sign() *NDArray {
	return unaryOp(a, func(x float64) float64 {
		switch {
		case math.IsNaN(x):
			return math.NaN()
		case x > 0:
			return 1
		case x < 0:
			return -1
		default:
			return 0
		}
	})
}

// Square returns the element-wise square of a (x*x).
func (a *NDArray) Square() *NDArray {
	return unaryOp(a, func(x float64) float64 { return x * x })
}

// Reciprocal returns the element-wise reciprocal 1/x of a.
func (a *NDArray) Reciprocal() *NDArray {
	return unaryOp(a, func(x float64) float64 { return 1 / x })
}

// Cbrt returns the element-wise cube root of a.
func (a *NDArray) Cbrt() *NDArray { return unaryOp(a, math.Cbrt) }

// Log2 returns the element-wise base-2 logarithm of a.
func (a *NDArray) Log2() *NDArray { return unaryOp(a, math.Log2) }

// Log10 returns the element-wise base-10 logarithm of a.
func (a *NDArray) Log10() *NDArray { return unaryOp(a, math.Log10) }

// Log1p returns the element-wise natural logarithm of 1+x, accurate for x near
// zero.
func (a *NDArray) Log1p() *NDArray { return unaryOp(a, math.Log1p) }

// Expm1 returns the element-wise value of exp(x)-1, accurate for x near zero.
func (a *NDArray) Expm1() *NDArray { return unaryOp(a, math.Expm1) }

// Clip returns a copy of a with every element limited to the range [min, max].
// Values below min become min and values above max become max. It panics if
// min > max.
func (a *NDArray) Clip(min, max float64) *NDArray {
	if min > max {
		panic("numpy: Clip requires min <= max")
	}
	return unaryOp(a, func(x float64) float64 {
		if x < min {
			return min
		}
		if x > max {
			return max
		}
		return x
	})
}
