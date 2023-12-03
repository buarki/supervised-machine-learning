package activation_test

import (
	"math"
	"testing"

	"github.com/buarki/supervised-machine-learning/activation"
)

const (
	acceptedError = 1e-8
)

func TestSigmoid(t *testing.T) {
	testCases := []struct {
		input    float64
		expected float64
	}{
		{input: 0, expected: 0.5},
		{input: 1, expected: math.E / (math.E + 1)},
		{input: 2, expected: (math.E * math.E) / (math.E*math.E + 1)},
		{input: -7, expected: 1 / (1 + math.Pow(math.E, 7))},
	}
	for _, testCase := range testCases {
		result := activation.Sigmoid(testCase.input)
		if math.Abs(testCase.expected-result) > acceptedError {
			t.Errorf("expected diff between sigmoid of %.8f - %.8f to be <= %.8f, got %.8f", testCase.input, testCase.expected, acceptedError, result)
		}
	}
}

func TestSigmoidPrime(t *testing.T) {
	testCases := []struct {
		input    float64
		expected float64
	}{
		{input: 0, expected: 0.25},
		{input: 1, expected: math.E / math.Pow((1+math.E), 2)},
		{input: -13, expected: 2.260319188837644e-06},
	}
	for _, testCase := range testCases {
		result := activation.SigmoidPrime(testCase.input)
		if math.Abs(testCase.expected-result) > acceptedError {
			t.Errorf("expected diff between sigmoid of %.8f - %.8f to be <= %.8f, got %.8f", testCase.input, testCase.expected, acceptedError, result)
		}
	}
}
