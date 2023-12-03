package neuralnet

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/buarki/supervised-machine-learning/matrix"
)

// Defining hyperparameters as inner constants
const (
	inputLayerSize      = 2
	outputLayerSize     = 1
	hiddenLayerSize     = 3
	amountOfInputParams = 3
)

// NeuralNet defines the data needed for the planned
// neural network. It is supposed to have an input layer
// with two inputs, a hidden layer with three neurons
// and an output layer with one ouput.
type NeuralNet struct {
	inputLayerSize       int     // The dimensions of input layer
	outputLayerSize      int     // How many neurons are present on second layer
	hiddenLayerSize      int     // The dimensions of output layer
	amountOfInputParams  int     // The number of input samples. Needed for normalization
	learningRate         float64 // Learning rate
	regularizationFactor float64 // Regularization factor

	w2 *matrix.Matrix // Matrix with weights of second layer
	b2 *matrix.Matrix // Matrix with bias values of second layer
	w3 *matrix.Matrix // Matrix with weights of third layer
	b3 *matrix.Matrix // Matrix with bias values of third layer

	activationFunction      func(v float64) float64 // Activation function to be used
	activationFunctionPrime func(v float64) float64 // Prime of the activation function used
}

// New creates and returns a neural network. It requires as argument the learning rate,
// regularization factor, activation function and the activation function prime.
func New(learningRate, regularizationFactor float64, activationFunction, activationFunctionPrime func(v float64) float64) (*NeuralNet, error) {
	// w2 is the second layer weights matrix. As it holds the weighs that will interact
	// with X it needs to be (2x3)
	w2, err := matrix.New(inputLayerSize, hiddenLayerSize, generateRandomValues(inputLayerSize*hiddenLayerSize))
	if err != nil {
		return nil, fmt.Errorf("failed to generate w2 weights, got %v", err)
	}
	// b2 is the second layer bias. As we sum it with v2 it must have the same dimension (3x3)
	b2, err := matrix.New(hiddenLayerSize, hiddenLayerSize, generateRandomValues(hiddenLayerSize*hiddenLayerSize))
	if err != nil {
		return nil, fmt.Errorf("failed to generate b2 weights, got %v", err)
	}
	// w3 is the second layer weights matrix. As it holds the weighs that will interact
	// with Y^2 it needs to be (3x1)
	w3, err := matrix.New(hiddenLayerSize, outputLayerSize, generateRandomValues(hiddenLayerSize*outputLayerSize))
	if err != nil {
		return nil, fmt.Errorf("failed to generate b2 weights, got %v", err)
	}
	// b3 is the third layer bias. As we sum it with v3 it must have the same dimension (3x1)
	b3, err := matrix.New(hiddenLayerSize, outputLayerSize, generateRandomValues(hiddenLayerSize*outputLayerSize))
	if err != nil {
		return nil, fmt.Errorf("failed to generate b2 weights, got %v", err)
	}
	return &NeuralNet{
		learningRate:            learningRate,
		regularizationFactor:    regularizationFactor,
		inputLayerSize:          inputLayerSize,
		outputLayerSize:         outputLayerSize,
		hiddenLayerSize:         hiddenLayerSize,
		amountOfInputParams:     amountOfInputParams,
		w2:                      w2,
		b2:                      b2,
		w3:                      w3,
		b3:                      b3,
		activationFunction:      activationFunction,
		activationFunctionPrime: activationFunctionPrime,
	}, nil
}

func (nn *NeuralNet) InputLayerSize() int {
	return nn.inputLayerSize
}

func (nn *NeuralNet) HiddenLayerSize() int {
	return nn.hiddenLayerSize
}

func (nn *NeuralNet) OutputLayerSize() int {
	return nn.outputLayerSize
}

// AdjustWeights changes w2 and w3 matrix. If given matrices
// have invalid shape it will error.
func (nn *NeuralNet) AdjustWeights(w2, w3 *matrix.Matrix) error {
	if w2 == nil {
		return fmt.Errorf("w2 cannot be nil")
	}
	if w3 == nil {
		return fmt.Errorf("w3 cannot be nil")
	}
	if nn.w2.Rows != w2.Rows || nn.w2.Columns != w2.Columns {
		return fmt.Errorf("given w2 has different shape, expected (%dx%d), received (%dx%d)", nn.w2.Rows, nn.w2.Columns, w2.Rows, w2.Columns)
	}
	if nn.w3.Rows != w3.Rows || nn.w3.Columns != w3.Columns {
		return fmt.Errorf("given w3 has different shape, expected (%dx%d), received (%dx%d)", nn.w3.Rows, nn.w3.Columns, w3.Rows, w3.Columns)
	}
	nn.w2 = w2
	nn.w3 = w3
	return nil
}

// AdjustBiases changes b2 and b3 matrix. If given matrices
// have invalid shape it will error.
func (nn *NeuralNet) AdjustBiases(b2, b3 *matrix.Matrix) error {
	if b2 == nil {
		return fmt.Errorf("b2 cannot be nil")
	}
	if b3 == nil {
		return fmt.Errorf("b3 cannot be nil")
	}
	if nn.b2.Rows != b2.Rows || nn.b2.Columns != b2.Columns {
		return fmt.Errorf("given b2 has different shape, expected (%dx%d), received (%dx%d)", nn.b2.Rows, nn.b2.Columns, b2.Rows, b2.Columns)
	}
	if nn.b3.Rows != b3.Rows || nn.b3.Columns != b3.Columns {
		return fmt.Errorf("given b3 has different shape, expected (%dx%d), received (%dx%d)", nn.b3.Rows, nn.b3.Columns, b3.Rows, b3.Columns)
	}
	nn.b2 = b2
	nn.b3 = b3
	return nil
}

func (nn *NeuralNet) RegularizationFactor() float64 {
	return nn.regularizationFactor
}

func (nn *NeuralNet) W2() *matrix.Matrix {
	return nn.w2
}

func (nn *NeuralNet) W3() *matrix.Matrix {
	return nn.w3
}

func (nn *NeuralNet) B2() *matrix.Matrix {
	return nn.b2
}

func (nn *NeuralNet) B3() *matrix.Matrix {
	return nn.b3
}

// Predict executes the forward process and returns the
// predicted values.
func (nn *NeuralNet) PredictBasedOn(X *matrix.Matrix) (*matrix.Matrix, error) {
	augmentedResult, err := nn.PredictForAnalysisBasedOn(X)
	if err != nil {
		return nil, err
	}
	return augmentedResult.Y3, nil
}

// ForwardResult is used in the training process to carry computed matrices.
type ForwardResult struct {
	W2 *matrix.Matrix
	B2 *matrix.Matrix
	W3 *matrix.Matrix
	B3 *matrix.Matrix
	V2 *matrix.Matrix
	Y2 *matrix.Matrix
	V3 *matrix.Matrix
	Y3 *matrix.Matrix
	X  *matrix.Matrix
}

type EvaluationResult struct {
	ErrorCost float64
	Error     *matrix.Matrix
}

func (nn *NeuralNet) Evaluate(expected, predicted *matrix.Matrix) (*EvaluationResult, error) {
	predictionErrorMatrix, err := nn.computeExpectedMinusPredicted(expected, predicted)
	if err != nil {
		return nil, err
	}
	errorCost, err := nn.computeErrorCost(predictionErrorMatrix)
	if err != nil {
		return nil, err
	}
	return &EvaluationResult{
		ErrorCost: errorCost,
		Error:     predictionErrorMatrix,
	}, nil
}

// PredictForAnalysisBasedOn executes the forward process and returns the computed
// matrices related to the backward process.
func (nn *NeuralNet) PredictForAnalysisBasedOn(X *matrix.Matrix) (*ForwardResult, error) {
	if X == nil {
		return nil, errors.New("param x cannot be nil")
	}

	v2, err := X.DotProductWith(nn.w2)
	if err != nil {
		return nil, fmt.Errorf("failed to compute v2, got %v", err)
	}
	v2PlusB2, err := v2.SumWith(nn.b2)
	if err != nil {
		return nil, fmt.Errorf("failed to compute x*w2 + b2, got %v", err)
	}
	y2, err := v2PlusB2.ApplyElementWise(nn.activationFunction)
	if err != nil {
		return nil, fmt.Errorf("failed to compute y2, got %v", err)
	}

	v3, err := y2.DotProductWith(nn.w3)
	if err != nil {
		return nil, fmt.Errorf("failed to compute v3, got %v", err)
	}
	v3PlusB3, err := v3.SumWith(nn.b3)
	if err != nil {
		return nil, fmt.Errorf("failed to compute y2*w3 + B3, got %v", err)
	}
	y3, err := v3PlusB3.ApplyElementWise(nn.activationFunction)
	if err != nil {
		return nil, fmt.Errorf("failed to compute y3, got %v", err)
	}

	return &ForwardResult{
		V2: v2PlusB2,
		Y2: y2,
		V3: v3PlusB3,
		Y3: y3,
		W2: nn.w2,
		W3: nn.w3,
		B2: nn.b2,
		B3: nn.b3,
		X:  X,
	}, nil
}

type GradientComponents struct {
	DEdW3 *matrix.Matrix
	DEdW2 *matrix.Matrix
	DEdB2 *matrix.Matrix
	DEdB3 *matrix.Matrix
}

// ComputeGradients computes the gradient descent components
// of w2,w3,b2 and b3.
func (nn *NeuralNet) ComputeGradients(expected, errorMatrix *matrix.Matrix, forwardResult *ForwardResult) (*GradientComponents, error) {
	res, err := nn.ComputeGradientsForAnalysis(expected, errorMatrix, forwardResult)
	if err != nil {
		return nil, err
	}
	return &GradientComponents{
		DEdW3: res.DEdW3,
		DEdW2: res.DEdW2,
		DEdB3: res.DEdB3,
		DEdB2: res.DEdB2,
	}, nil
}

type AugmentedGradientComponents struct {
	DEdW3  *matrix.Matrix
	DEdW2  *matrix.Matrix
	DEdB2  *matrix.Matrix
	DEdB3  *matrix.Matrix
	Delta3 *matrix.Matrix
	Delta2 *matrix.Matrix
	W2     *matrix.Matrix
	W3     *matrix.Matrix
	X      *matrix.Matrix
	B2     *matrix.Matrix
	B3     *matrix.Matrix
}

func (nn *NeuralNet) ComputeGradientsForAnalysis(expected, errorMatrix *matrix.Matrix, forwardResult *ForwardResult) (*AugmentedGradientComponents, error) {
	delta3, dEdW3, dEdB3, err := nn.computeLayer3Params(expected, forwardResult)
	if err != nil {
		return nil, fmt.Errorf("failed to compute layer 3 params, got %v", err)
	}
	delta2, dEdW2, dEdB2, err := nn.computeLayer2Params(delta3, expected, forwardResult)
	if err != nil {
		return nil, fmt.Errorf("failed to compute layer 2 params, got %v", err)
	}
	return &AugmentedGradientComponents{
		DEdW3:  dEdW3,
		DEdW2:  dEdW2,
		DEdB3:  dEdB3,
		DEdB2:  dEdB2,
		Delta3: delta3,
		Delta2: delta2,
		W2:     nn.w2,
		W3:     nn.w3,
		X:      forwardResult.X,
	}, nil
}

func (nn *NeuralNet) computeLayer3Params(expected *matrix.Matrix, forwardResult *ForwardResult) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, error) {
	errorMatrix, err := nn.computeExpectedMinusPredicted(expected, forwardResult.Y3)
	if err != nil {
		return nil, nil, nil, err
	}
	delta3, err := nn.computeDelta3(errorMatrix, forwardResult.V3)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to compute delta3, got %v", err)
	}
	dEdW3, err := nn.computeDdEdW3(delta3, forwardResult.Y2, forwardResult.X)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("faile to compute dEdW3, got %v", err)
	}
	dEdB3, err := nn.computeDEdB3(delta3, forwardResult.X)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to compute dEdB3, got %v", err)
	}
	return delta3, dEdW3, dEdB3, nil
}

func (nn *NeuralNet) computeDelta3(errorMatrix, V3 *matrix.Matrix) (*matrix.Matrix, error) {
	sigmoidPrimeOfV3, err := V3.ApplyElementWise(nn.activationFunctionPrime)
	if err != nil {
		return nil, fmt.Errorf("failed to compute sigmoid prime of v3, got %v", err)
	}
	hadamardOfErrorMatrixAndSigmoidPrime, err := errorMatrix.HadamardProductWith(sigmoidPrimeOfV3)
	if err != nil {
		return nil, fmt.Errorf("failed to compute hadamard product of (expected - predicted) * sigmoid prime of v3, got %v", err)
	}
	delta3, err := hadamardOfErrorMatrixAndSigmoidPrime.ApplyElementWise(func(value float64) float64 {
		return -value
	})
	if err != nil {
		return nil, fmt.Errorf("failed to compute delta3, got %v", err)
	}
	return delta3, nil
}

func (nn *NeuralNet) computeDdEdW3(delta3, Y2, X *matrix.Matrix) (*matrix.Matrix, error) {
	dEdW3, err := Y2.T().DotProductWith(delta3)
	if err != nil {
		return nil, fmt.Errorf("failed to compute dEdW3, got %v", err)
	}
	dEdW3Normalized, err := dEdW3.ApplyElementWise(func(value float64) float64 {
		return value / float64(nn.amountOfInputParams)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to normalize dEdW3, got %v", err)
	}
	dEdW3Penalty, err := nn.w3.ApplyElementWise(func(value float64) float64 {
		return value * nn.regularizationFactor
	})
	if err != nil {
		return nil, fmt.Errorf("failed to compute penalty of dEdW3, got %v", err)
	}
	dEdW3Regularized, err := dEdW3Normalized.SumWith(dEdW3Penalty)
	if err != nil {
		return nil, fmt.Errorf("failed to compute penalty of dEdW3 + penalty, got %v", err)
	}
	return dEdW3Regularized, nil
}

func (nn *NeuralNet) computeDEdB3(delta3, X *matrix.Matrix) (*matrix.Matrix, error) {
	delta3Normalized, err := delta3.ApplyElementWise(func(value float64) float64 {
		return value / float64(nn.amountOfInputParams)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to normalize delta3, got %v", err)
	}
	delta3Penalty, err := nn.b3.ApplyElementWise(func(value float64) float64 {
		return value * nn.regularizationFactor
	})
	if err != nil {
		return nil, fmt.Errorf("failed to compute penalty for delta3, got %v", err)
	}
	dEdB3, err := delta3Normalized.SumWith(delta3Penalty)
	if err != nil {
		return nil, fmt.Errorf("failed to compute normalized delta3 + penalty, got %v", err)
	}
	return dEdB3, nil
}

func (nn *NeuralNet) computeLayer2Params(delta3, expected *matrix.Matrix, forwardResult *ForwardResult) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix, error) {
	delta2, err := nn.computeDelta2(delta3, forwardResult.V2)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to compute delta2, got %v", err)
	}
	dEdW2, err := nn.computeDEdW2(delta2, forwardResult.X)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to compute penalty of dEdW2 + penalty, got %v", err)
	}
	dEdB2, err := nn.computeDEdB2(delta2)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to compute dEdB2, got %v", err)
	}
	return delta2, dEdW2, dEdB2, nil
}

func (nn *NeuralNet) computeDelta2(delta3, V2 *matrix.Matrix) (*matrix.Matrix, error) {
	delta3TimesW3T, err := delta3.DotProductWith(nn.w3.T())
	if err != nil {
		return nil, fmt.Errorf("failed to compute delta3 dot with W3T, got %v", err)
	}
	sigmoidPrimeOfV2, err := V2.ApplyElementWise(nn.activationFunctionPrime)
	if err != nil {
		return nil, fmt.Errorf("failed to compute sigmoid prime of v2, got %v", err)
	}
	delta2, err := delta3TimesW3T.HadamardProductWith(sigmoidPrimeOfV2)
	if err != nil {
		return nil, fmt.Errorf("failed to compute delta3*W3T*sigmoidPrime of v2, got %v", err)
	}
	return delta2, nil
}

func (nn *NeuralNet) computeDEdW2(delta2, X *matrix.Matrix) (*matrix.Matrix, error) {
	dEdW2, err := X.T().DotProductWith(delta2)
	if err != nil {
		return nil, fmt.Errorf("failed to compute dEdW2, got %v", err)
	}
	dEdW2Normalized, err := dEdW2.ApplyElementWise(func(value float64) float64 {
		return value / float64(nn.amountOfInputParams)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to normalize dEdW2, got %v", err)
	}
	dEdW2Penalty, err := nn.w2.ApplyElementWise(func(value float64) float64 {
		return value * nn.regularizationFactor
	})
	if err != nil {
		return nil, fmt.Errorf("failed to compute penalty of dEdW2, got %v", err)
	}
	dEdW2Regularized, err := dEdW2Normalized.SumWith(dEdW2Penalty)
	if err != nil {
		return nil, fmt.Errorf("failed to compute penalty of dEdW2 + penalty, got %v", err)
	}
	return dEdW2Regularized, nil
}

func (nn *NeuralNet) computeDEdB2(delta2 *matrix.Matrix) (*matrix.Matrix, error) {
	delta2Normalized, err := delta2.ApplyElementWise(func(value float64) float64 {
		return value / float64(nn.amountOfInputParams)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to normalize delta2, got %v", err)
	}
	delta2Penalty, err := nn.b2.ApplyElementWise(func(value float64) float64 {
		return value * nn.regularizationFactor
	})
	if err != nil {
		return nil, fmt.Errorf("failed to compute penalty for delta2, got %v", err)
	}
	dEdB2, err := delta2Normalized.SumWith(delta2Penalty)
	if err != nil {
		return nil, fmt.Errorf("failed to compute normalized delta2 + penalty, got %v", err)
	}
	return dEdB2, nil
}

type PredictionWithError struct {
	Prediction *matrix.Matrix
	Error      float64
}

func (nn *NeuralNet) computeErrorCost(errorMatrix *matrix.Matrix) (float64, error) {
	w2Hadamard, err := nn.W2().HadamardProductWith(nn.W2())
	if err != nil {
		return 0, fmt.Errorf("failed to compute hadamard product w2 * w2, got %v", err)
	}
	w3Hadamard, err := nn.W3().HadamardProductWith(nn.W3())
	if err != nil {
		return 0, fmt.Errorf("failed to compute hadamard product w3 * w3, got %v", err)
	}
	penalty := (nn.RegularizationFactor() / 2.0) * (w2Hadamard.SumOfAllElements() + w3Hadamard.SumOfAllElements())
	errorMatrixHadamardProduct, err := errorMatrix.HadamardProductWith(errorMatrix)
	if err != nil {
		return 0, fmt.Errorf("failed to compute hadamard product (expected - predicted)*(expected - predicted), got %v", err)
	}
	return (0.5 * errorMatrixHadamardProduct.SumOfAllElements() / float64(nn.amountOfInputParams)) + penalty, nil
}

func (nn *NeuralNet) computeExpectedMinusPredicted(expected, predicted *matrix.Matrix) (*matrix.Matrix, error) {
	predictionError, err := expected.Minus(predicted)
	if err != nil {
		return nil, fmt.Errorf("failed to compute prediction error, got %v", err)
	}
	return predictionError, nil
}

// ToJSON can be used to export the neural net state.
func (nn *NeuralNet) ToJSON() (string, error) {
	neuralNetState := struct {
		LearningRate         float64   `json:"learningRate"`
		RegularizationFactor float64   `json:"regularizationFactor"`
		W2                   []float64 `json:"w2"`
		W3                   []float64 `json:"w3"`
		B2                   []float64 `json:"b2"`
		B3                   []float64 `json:"b3"`
	}{
		W2:                   nn.w2.FlattenedElements(),
		W3:                   nn.w3.FlattenedElements(),
		B2:                   nn.b2.FlattenedElements(),
		B3:                   nn.b3.FlattenedElements(),
		LearningRate:         nn.learningRate,
		RegularizationFactor: nn.regularizationFactor,
	}
	b, err := json.Marshal(neuralNetState)
	if err != nil {
		return "", fmt.Errorf("failed to generate neural net JSON, got %v", err)
	}
	return string(b), nil
}
