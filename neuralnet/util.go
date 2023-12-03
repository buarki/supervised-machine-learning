package neuralnet

import (
	"math"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func generateRandomValues(amountOfValues int) []float64 {
	randomValues := make([]float64, amountOfValues)
	for i := 0; i < amountOfValues; i++ {
		randomValues[i] = randomNonZeroValue()
	}
	return randomValues
}

func randomNonZeroValue() float64 {
	const epsilon = 1e-9
	value := rand.Float64()*2 - 1
	for math.Abs(value) < epsilon {
		value = rand.Float64()*2 - 1
	}
	return value
}
