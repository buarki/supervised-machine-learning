package activation

import "math"

// Sigmoid implements the sigmoid function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

// SigmoidPrime implements the derivative of a sigmoid function
func SigmoidPrime(x float64) float64 {
	return math.Pow(math.E, -x) / math.Pow((1+math.Pow(math.E, -x)), 2)
}
