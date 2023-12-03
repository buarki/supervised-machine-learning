package neuralnet_test

import (
	"fmt"
	"testing"

	"github.com/buarki/supervised-machine-learning/activation"
	"github.com/buarki/supervised-machine-learning/matrix"
	"github.com/buarki/supervised-machine-learning/neuralnet"
	"github.com/buarki/supervised-machine-learning/sample"
)

/*
This test is a sanity test just to check the full
forward process by checking the result of computed
v2,y2,w3 and y3.
*/
func TestExecutePredictBasedOn(t *testing.T) {
	inputOutput, err := sample.GetAReadyInputAndOutputSample()
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	X := inputOutput.Input
	nn, err := neuralnet.New(0.001, 0.0001, activation.Sigmoid, activation.SigmoidPrime)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	weightsSample, err := sample.GetWeights()
	if err != nil {
		t.Errorf("expected to be nil, got %v", err)
	}
	sampleBiases, err := sample.GetBiases()
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	knownB2 := sampleBiases.B2
	knownB3 := sampleBiases.B3
	knownW2 := weightsSample.W2
	knownW3 := weightsSample.W3
	expectedV2, err := matrix.New(3, 3, []float64{
		0.792667185, 1.76936391, -0.291706973,
		-0.917605334, -0.0777543792, 0.654353841,
		-0.900988121, -0.0495140693, -0.970005329,
	})
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	expectedY2, err := matrix.New(3, 3, []float64{
		0.68840374, 0.85437855, 0.42758602,
		0.28544608, 0.48057119, 0.65799092,
		0.28884748, 0.48762401, 0.27487944,
	})
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	expectedV3, err := matrix.New(3, 1, []float64{
		-0.71246648,
		0.25750816,
		-1.63416412,
	})
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	expectedY3, err := matrix.New(3, 1, []float64{
		0.32905407,
		0.56402364,
		0.16326072,
	})
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	if err := nn.AdjustWeights(knownW2, knownW3); err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if err := nn.AdjustBiases(knownB2, knownB3); err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	prediction, err := nn.PredictForAnalysisBasedOn(X)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	ensureMatricesAreEqual(t, prediction.W2, knownW2)
	ensureMatricesAreEqual(t, prediction.W3, knownW3)
	ensureMatricesAreEqual(t, prediction.B2, knownB2)
	ensureMatricesAreEqual(t, prediction.B3, knownB3)
	ensureMatricesMatch(t, prediction.V2, expectedV2, fmt.Sprintf("received V2 does not match with expected. Received V2:\n%s\nExpected V2:\n%s\n", prediction.V2.ToString(), expectedV2.ToString()))
	ensureMatricesMatch(t, prediction.Y2, expectedY2, fmt.Sprintf("received Y2 does not match with expected. Received Y2:\n%s\nExpected Y2:\n%s\n", prediction.Y2.ToString(), expectedY2.ToString()))
	ensureMatricesMatch(t, prediction.V3, expectedV3, fmt.Sprintf("received V3 does not match with expected. Received V3:\n%s\nExpected V3:\n%s\n", prediction.V3.ToString(), expectedV3.ToString()))
	ensureMatricesMatch(t, prediction.Y3, expectedY3, fmt.Sprintf("received Y3 does not match with expected. Received Y3:\n%s\nExpected Y3:\n%s\n", prediction.Y3.ToString(), expectedY3.ToString()))
}

func TestAdjustWeightsWithInvalidW2(t *testing.T) {
	w2MatrixWithWrongDimensions, err := matrix.New(1, 1, []float64{1})
	if err != nil {
		t.Errorf("failed to create wrong w2, got %v", err)
	}
	nn, err := neuralnet.New(0.001, 0.0001, activation.Sigmoid, activation.SigmoidPrime)
	if err != nil {
		t.Errorf("failed to create nn, got %v", err)
	}
	if err := nn.AdjustWeights(w2MatrixWithWrongDimensions, nn.W3()); err == nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
}

func TestAdjustWeightsWithInvalidW3(t *testing.T) {
	w3MatrixWithWrongDimensions, err := matrix.New(1, 1, []float64{1})
	if err != nil {
		t.Errorf("failed to create wrong w3, got %v", err)
	}
	nn, err := neuralnet.New(0.001, 0.0001, activation.Sigmoid, activation.SigmoidPrime)
	if err != nil {
		t.Errorf("failed to create nn, got %v", err)
	}
	if err := nn.AdjustWeights(nn.W2(), w3MatrixWithWrongDimensions); err == nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
}

func TestToJSON(t *testing.T) {
	expectedLearningRate := 0.001
	expectedRegularizationFactor := 0.0001

	nn, err := neuralnet.New(expectedLearningRate, expectedRegularizationFactor, activation.Sigmoid, activation.SigmoidPrime)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	w2, err := matrix.New(2, 3, []float64{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Errorf("failed to create w2, got %v", err)
	}

	w3, err := matrix.New(3, 1, []float64{7, 8, 9})
	if err != nil {
		t.Errorf("failed to create w2, got %v", err)
	}

	b2, err := matrix.New(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	if err != nil {
		t.Errorf("failed to create b2, got %v", err)
	}

	b3, err := matrix.New(3, 1, []float64{7, 8, 9})
	if err != nil {
		t.Errorf("failed to create b3, got %v", err)
	}

	if err := nn.AdjustWeights(w2, w3); err != nil {
		t.Errorf("failed to adjust weights, got %v", err)
	}

	if err := nn.AdjustBiases(b2, b3); err != nil {
		t.Errorf("failed to adjust biases, got %v", err)
	}

	nnJson, err := nn.ToJSON()
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	expectedJson := `{"learningRate":0.001,"regularizationFactor":0.0001,"w2":[1,2,3,4,5,6],"w3":[7,8,9],"b2":[1,2,3,4,5,6,7,8,9],"b3":[7,8,9]}`
	if nnJson != expectedJson {
		t.Errorf("expected received json to be %s, got %s", expectedJson, nnJson)
	}
}

func TestGradientComponentsCalculationWithKnownInputOutputAndWeights_1(t *testing.T) {
	sampleWeigths, err := sample.GetWeights()
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	knownW2 := sampleWeigths.W2
	knownW3 := sampleWeigths.W3
	sampleBiases, err := sample.GetBiases()
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	knownB2 := sampleBiases.B2
	knownB3 := sampleBiases.B3
	inputOutput, err := sample.GetAReadyInputAndOutputSample()
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	X := inputOutput.Input
	Y := inputOutput.Output
	nn, err := neuralnet.New(0.001, 0.0001, activation.Sigmoid, activation.SigmoidPrime)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	expectedDelta3, err := matrix.New(3, 1, []float64{
		-1.5831834317580953,
		-1.8776940167768534,
		-1.2481393907717802,
	})
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	expectedDEdW3, err := matrix.New(3, 1, []float64{
		-0.6620802075143511,
		-0.9546149779996012,
		-0.7519211767397317,
	})
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	expectedDelta2, err := matrix.New(3, 3, []float64{
		-0.14838073302058022, 0.14351516015363522, 0.28797754673577824,
		-0.16733805844595845, 0.34150686759302057, 0.3140335299621354,
		-0.11202249593379544, 0.2272101849555621, 0.18488821406208106,
	})
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	expectedDEdW2, err := matrix.New(2, 3, []float64{
		-0.08006490486504064, 0.147088658404093, 0.14266884536413632,
		-0.0755568160487767, 0.10098182275802815, 0.14166393392825902,
	})
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if err := nn.AdjustWeights(knownW2, knownW3); err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	if err := nn.AdjustBiases(knownB2, knownB3); err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	forwardResult, err := nn.PredictForAnalysisBasedOn(X)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	evaluationResult, err := nn.Evaluate(Y, forwardResult.Y3)
	if err != nil {
		t.Errorf("expected error nil, got %v", err)
	}
	result, err := nn.ComputeGradientsForAnalysis(Y, evaluationResult.Error, forwardResult)
	if err != nil {
		t.Errorf("expected error nil, got %v", err)
	}

	ensureMatricesAreEqual(t, result.W2, knownW2)
	ensureMatricesAreEqual(t, result.W3, knownW3)
	ensureMatricesAreEqual(t, result.X, X)
	ensureMatricesMatch(t, result.Delta3, expectedDelta3, fmt.Sprintf("received Delta3 does not match with expected. Received delta3:\n%s\nExpected delta3:\n%s\n", result.Delta3.ToString(), expectedDelta3.ToString()))
	ensureMatricesMatch(t, result.DEdW3, expectedDEdW3, fmt.Sprintf("received dEdW3 does not match with expected. Received dEdW3:\n%s\nExpected dEdW3:\n%s\n", result.DEdW3.ToString(), expectedDEdW3.ToString()))
	ensureMatricesMatch(t, result.Delta2, expectedDelta2, fmt.Sprintf("received Delta2 does not match with expected. Received delta2:\n%s\nExpected delta2:\n%s\n", result.Delta2.ToString(), expectedDelta2.ToString()))
	ensureMatricesMatch(t, result.DEdW2, expectedDEdW2, fmt.Sprintf("received dEdW2 does not match with expected. Received dEdW2:\n%s\nExpected dEdW2:\n%s\n", result.DEdW2.ToString(), expectedDEdW2.ToString()))
}

func TestGradientDescentAccurace_1(t *testing.T) {
	sample, err := sample.GetAReadyInputAndOutputSample()
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	X := sample.Input
	Y := sample.Output

	nn, err := neuralnet.New(0.001, 0.0001, activation.Sigmoid, activation.SigmoidPrime)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	forwardResult, err := nn.PredictForAnalysisBasedOn(X)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	evaluation, err := nn.Evaluate(Y, forwardResult.Y3)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	gradientComponents, err := nn.ComputeGradients(Y, evaluation.Error, forwardResult)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	defaultGradients := flattenParams(gradientComponents.DEdW2, gradientComponents.DEdW3, gradientComponents.DEdB2, gradientComponents.DEdB3)

	numericalGradients, err := getNumericalGradient(nn, X, Y)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	diff := normOfSlice(subtractArrays(defaultGradients, numericalGradients)) / normOfSlice(sumArrays(defaultGradients, numericalGradients))
	acceptedError := 1e-4
	if diff >= acceptedError {
		t.Errorf("expected diff to be < %v, got %v", acceptedError, diff)
	}
}
