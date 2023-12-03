package data

import (
	"github.com/buarki/supervised-machine-learning/matrix"
	"github.com/buarki/supervised-machine-learning/neuralnet"
)

// Transform transform a sample into training data
func Transform(samples []Sample) ([]neuralnet.TrainingData, error) {
	var trainingData []neuralnet.TrainingData
	for i := 0; i < len(samples); i += 3 {
		triplet := samples[i : i+3]
		var xData []float64
		var yData []float64
		for _, td := range triplet {
			xData = append(xData, float64(td.HoursOfSleep), float64(td.HoursOfMeditation))
			yData = append(yData, float64(td.ScoreTest))
		}
		X, err := matrix.New(3, 2, xData)
		if err != nil {
			panic(err)
		}
		Y, err := matrix.New(3, 1, yData)
		if err != nil {
			panic(err)
		}
		trainingData = append(trainingData, neuralnet.TrainingData{X: X, Y: Y})
	}
	return trainingData, nil
}
