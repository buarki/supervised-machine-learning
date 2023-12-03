package data

import (
	"fmt"

	"github.com/buarki/supervised-machine-learning/neuralnet"
	"github.com/buarki/supervised-machine-learning/normalize"
)

// Normalize normalize received training data
func Normalize(samples []neuralnet.TrainingData) ([]neuralnet.TrainingData, error) {
	var normalized []neuralnet.TrainingData
	for _, sample := range samples {
		xNormalized, err := normalize.Input(sample.X)
		if err != nil {
			return nil, fmt.Errorf("failed to normalized X, got %v", err)
		}
		yNormalized, err := normalize.Output(sample.Y)
		if err != nil {
			return nil, fmt.Errorf("failed to normalized Y, got %v", err)
		}
		normalized = append(normalized, neuralnet.TrainingData{X: xNormalized, Y: yNormalized})
	}
	return normalized, nil
}
