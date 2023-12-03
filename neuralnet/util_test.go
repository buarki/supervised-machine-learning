package neuralnet_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/buarki/supervised-machine-learning/matrix"
	"github.com/buarki/supervised-machine-learning/neuralnet"
)

const (
	acceptedError = 1e-7
)

func ensureMatricesAreEqual(t *testing.T, receivedW2, injectedW2 *matrix.Matrix) {
	diff, err := receivedW2.FrobeniusNormRatio(injectedW2)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	if diff != 0 {
		t.Errorf("expected that diff between received W2 and Injected was 0, got %v", diff)
	}
}

func ensureMatricesMatch(t *testing.T, received, expected *matrix.Matrix, errorMessage string) {
	diff, err := received.FrobeniusNormRatio(expected)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	if diff > acceptedError {
		t.Error(errorMessage)
	}
}

func normOfSlice(s []float64) float64 {
	sumOfSquares := 0.0
	for _, value := range s {
		sumOfSquares += value * value
	}
	return math.Sqrt(sumOfSquares)
}

func sumArrays(a, b []float64) []float64 {
	sum := make([]float64, len(a))
	for i := range a {
		sum[i] = a[i] + b[i]
	}
	return sum
}

func subtractArrays(a, b []float64) []float64 {
	sum := make([]float64, len(a))
	for i := range a {
		sum[i] = a[i] - b[i]
	}
	return sum
}

func reassembleParamsFromFlatArray(nn *neuralnet.NeuralNet, flatWeights []float64) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, *matrix.Matrix, error) {
	w2Begin := 0
	w2End := nn.InputLayerSize() * nn.HiddenLayerSize()

	w2WeightValues := flatWeights[w2Begin:w2End]
	w2, err := matrix.New(nn.InputLayerSize(), nn.HiddenLayerSize(), w2WeightValues)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to rebuild w2, got %v", err)
	}

	w3Begin := w2End
	w3End := w3Begin + (nn.HiddenLayerSize() * nn.OutputLayerSize())
	w3WeightValues := flatWeights[w3Begin:w3End]
	w3, err := matrix.New(nn.HiddenLayerSize(), nn.OutputLayerSize(), w3WeightValues)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to rebuild w3, got %v", err)
	}

	b2Begin := w3End
	b2End := b2Begin + (nn.HiddenLayerSize() * nn.HiddenLayerSize())
	b2Values := flatWeights[b2Begin:b2End]
	b2, err := matrix.New(nn.HiddenLayerSize(), nn.HiddenLayerSize(), b2Values)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to rebuild b2, got %v", err)
	}

	b3Begin := b2End
	b3End := b3Begin + (nn.HiddenLayerSize() * nn.OutputLayerSize())
	b3Values := flatWeights[b3Begin:b3End]
	b3, err := matrix.New(nn.HiddenLayerSize(), nn.OutputLayerSize(), b3Values)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to rebuild b3, got %v", err)
	}

	return w2, w3, b2, b3, nil
}

func flattenParams(w2, w3, b2, b3 *matrix.Matrix) []float64 {
	var flattenedNeuralNetWeights []float64
	flattenedNeuralNetWeights = append(flattenedNeuralNetWeights, w2.FlattenedElements()...)
	flattenedNeuralNetWeights = append(flattenedNeuralNetWeights, w3.FlattenedElements()...)
	flattenedNeuralNetWeights = append(flattenedNeuralNetWeights, b2.FlattenedElements()...)
	flattenedNeuralNetWeights = append(flattenedNeuralNetWeights, b3.FlattenedElements()...)
	return flattenedNeuralNetWeights
}

// computing the gradient for each weight mannualy
func getNumericalGradient(nn *neuralnet.NeuralNet, X, Y *matrix.Matrix) ([]float64, error) {
	flattenedNeuralNetParams := flattenParams(nn.W2(), nn.W3(), nn.B2(), nn.B3())
	numericalGradient := make([]float64, len(flattenedNeuralNetParams))
	pertub := make([]float64, len(flattenedNeuralNetParams))
	smallDifference := 1e-3
	for weightIndex := range flattenedNeuralNetParams {
		pertub[weightIndex] = smallDifference

		// checking the right
		w2, w3, b2, b3, err := reassembleParamsFromFlatArray(nn, sumArrays(flattenedNeuralNetParams, pertub))
		if err != nil {
			return nil, fmt.Errorf("failed to reassamble w2, w3, b2 and b3 to check values at right, got %v", err)
		}
		if err := nn.AdjustWeights(w2, w3); err != nil {
			return nil, fmt.Errorf("failed to adjust weights, got %v", err)
		}
		if err := nn.AdjustBiases(b2, b3); err != nil {
			return nil, fmt.Errorf("failed to adjust biases, got %v", err)
		}
		predictionForRight, err := nn.PredictBasedOn(X)
		if err != nil {
			return nil, fmt.Errorf("failed to predict Y for the right, got %v", err)
		}
		evaluationForRight, err := nn.Evaluate(Y, predictionForRight)
		if err != nil {
			return nil, fmt.Errorf("failed to evaluate result for right, got %v", err)
		}
		errorCostAtRight := evaluationForRight.ErrorCost

		// checking the left
		w2, w3, b2, b3, err = reassembleParamsFromFlatArray(nn, subtractArrays(flattenedNeuralNetParams, pertub))
		if err != nil {
			return nil, fmt.Errorf("failed to reassamble w2, w3, b2 and b3 to check values at left, got %v", err)
		}
		if err := nn.AdjustWeights(w2, w3); err != nil {
			return nil, fmt.Errorf("failed to adjust weights, got %v", err)
		}
		if err := nn.AdjustBiases(b2, b3); err != nil {
			return nil, fmt.Errorf("failed to adjust biases, got %v", err)
		}
		predictionForLeft, err := nn.PredictBasedOn(X)
		if err != nil {
			return nil, fmt.Errorf("failed to predict Y for the left, got %v", err)
		}
		evaluationForLeft, err := nn.Evaluate(Y, predictionForLeft)
		if err != nil {
			return nil, fmt.Errorf("ailed to evaluate result for left, got %v", err)
		}
		errorCostAtLeft := evaluationForLeft.ErrorCost

		numericalGradient[weightIndex] = (errorCostAtRight - errorCostAtLeft) / (2 * smallDifference)
		pertub[weightIndex] = 0.0
	}
	w2, w3, b2, b3, err := reassembleParamsFromFlatArray(nn, flattenedNeuralNetParams)
	if err != nil {
		return nil, fmt.Errorf("failed to rebuild original w2, w3, b2 and b3, got %v", err)
	}
	if err := nn.AdjustWeights(w2, w3); err != nil {
		return nil, fmt.Errorf("failed to adjust weights, got %v", err)
	}
	if err := nn.AdjustBiases(b2, b3); err != nil {
		return nil, fmt.Errorf("failed to adjust biases, got %v", err)
	}
	return numericalGradient, nil
}
