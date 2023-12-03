package main

import (
	"fmt"
	"log"

	"github.com/buarki/supervised-machine-learning/activation"
	"github.com/buarki/supervised-machine-learning/data"
	"github.com/buarki/supervised-machine-learning/neuralnet"
)

func main() {
	csvTrainingDataPath := "sample/sample.csv"
	samples, err := data.LoadDataSetFrom(csvTrainingDataPath)
	if err != nil {
		log.Fatalf("failed to load data set from path %s, got %v", csvTrainingDataPath, err)
	}
	trainingData, err := data.Transform(samples)
	if err != nil {
		log.Fatalf("failed to transform data, got %v", err)
	}
	normalized, err := data.Normalize(trainingData)
	if err != nil {
		log.Fatalf("failed to normalize data set, got %v", err)
	}

	sizeOfBatchToTrain := 40
	traningBatch := normalized[:sizeOfBatchToTrain]
	validationBatch := normalized[sizeOfBatchToTrain:]

	learningRate := 0.001
	regularizationFactor := 0.0001
	nn, err := neuralnet.New(learningRate, regularizationFactor, activation.Sigmoid, activation.SigmoidPrime)
	if err != nil {
		log.Fatalf("failed to create neural network, got %v", err)
	}

	trainingEpochs := 600_000
	if err := neuralnet.Train(nn, trainingEpochs, traningBatch); err != nil {
		log.Fatalf("failed to train neural net, got %v", err)
	}

	fmt.Println("===============")
	fmt.Println("===============")
	fmt.Println("===============")
	fmt.Println("===============")
	fmt.Println()
	fmt.Println("===============[Doing some checking]==============")
	fmt.Println("First check:")
	fmt.Println("X:")
	fmt.Println(validationBatch[3].X.ToString())
	p, err := nn.PredictBasedOn(validationBatch[0].X)
	if err != nil {
		log.Fatalf("failed to predict data, got %v", err)
	}
	fmt.Println("PREDICTED:")
	fmt.Println(p.ToString())
	fmt.Println("EXPECTED:")
	fmt.Println(validationBatch[0].Y.ToString())
	fmt.Println()
	fmt.Println("Second check:")
	fmt.Println("X:")
	fmt.Println(validationBatch[1].X.ToString())
	p, err = nn.PredictBasedOn(validationBatch[1].X)
	if err != nil {
		log.Fatalf("failed to predict data, got %v", err)
	}
	fmt.Println("PREDICTED:")
	fmt.Println(p.ToString())
	fmt.Println("EXPECTED:")
	fmt.Println(validationBatch[4].Y.ToString())

	fmt.Println("\n\nNeural state state:")
	nnJSON, err := nn.ToJSON()
	if err != nil {
		log.Fatalf("failed to get json of neural net, got %v", err)
	}
	fmt.Println(nnJSON)
}
