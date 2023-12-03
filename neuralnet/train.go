package neuralnet

import (
	"fmt"
	"log"

	"github.com/buarki/supervised-machine-learning/matrix"
)

type TrainingData struct {
	X *matrix.Matrix
	Y *matrix.Matrix
}

// Train trains a neural network by injecting data into it
// while iterating over the epochs.
func Train(nn *NeuralNet, epochs int, trainingData []TrainingData) error {
	for epoch := 0; epoch < epochs; epoch++ {
		log.Printf("starting epoch %d/%d\n", epoch+1, epochs)
		for trainingDataIndex, data := range trainingData {
			log.Printf("learning with training data... %d/%d\n", trainingDataIndex+1, len(trainingData))
			forwardResult, err := nn.PredictForAnalysisBasedOn(data.X)
			if err != nil {
				return err
			}
			evaluationError, err := nn.Evaluate(data.Y, forwardResult.Y3)
			if err != nil {
				return err
			}
			gradientComponents, err := nn.ComputeGradients(data.Y, evaluationError.Error, forwardResult)
			if err != nil {
				return err
			}
			if err != nil {
				return fmt.Errorf("failed to compute gradients, got %v", err)
			}
			newW2, err := computeNewParam(nn.learningRate, nn.W2(), gradientComponents.DEdW2)
			if err != nil {
				return fmt.Errorf("failed to compute new W2, got %v", err)
			}
			newW3, err := computeNewParam(nn.learningRate, nn.W3(), gradientComponents.DEdW3)
			if err != nil {
				return fmt.Errorf("failed to compute new W3, got %v", err)
			}
			if err := nn.AdjustWeights(newW2, newW3); err != nil {
				return fmt.Errorf("failed to adjust weights during train, got %v", err)
			}
			newB2, err := computeNewParam(nn.learningRate, nn.B2(), gradientComponents.DEdB2)
			if err != nil {
				return fmt.Errorf("failed to compute new B2, got %v", err)
			}
			newB3, err := computeNewParam(nn.learningRate, nn.B3(), gradientComponents.DEdB3)
			if err != nil {
				return fmt.Errorf("failed to compute new B3, got %v", err)
			}
			if err := nn.AdjustBiases(newB2, newB3); err != nil {
				return fmt.Errorf("failed to adjust biases during train, got %v", err)
			}
			log.Printf("learned using data %d/%d, got error %.7f\n", trainingDataIndex+1, len(trainingData), evaluationError.ErrorCost)
		}
		log.Printf("finished epoch %d\n", epoch+1)
	}
	return nil
}

func computeNewParam(learningRate float64, oldParam, paramGradientComponent *matrix.Matrix) (*matrix.Matrix, error) {
	paramGradientComponentTimesLearninRate, err := paramGradientComponent.ApplyElementWise(func(value float64) float64 {
		return value * learningRate
	})
	if err != nil {
		return nil, fmt.Errorf("failed to compute learning rate * weight, got %v", err)
	}
	newParam, err := oldParam.Minus(paramGradientComponentTimesLearninRate)
	if err != nil {
		return nil, fmt.Errorf("failed to compute weight - (learningRate*weight), got %v", err)
	}
	return newParam, nil
}
