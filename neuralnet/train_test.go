package neuralnet_test

import (
	"testing"

	"github.com/buarki/supervised-machine-learning/activation"
	"github.com/buarki/supervised-machine-learning/neuralnet"
	"github.com/buarki/supervised-machine-learning/sample"
)

func TestTrain(t *testing.T) {
	nn, err := neuralnet.New(0.001, 0.0001, activation.Sigmoid, activation.SigmoidPrime)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	sample, err := sample.GetAReadyInputAndOutputSample()
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	if err := neuralnet.Train(nn, 1, []neuralnet.TrainingData{{X: sample.Input, Y: sample.Output}}); err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
}
