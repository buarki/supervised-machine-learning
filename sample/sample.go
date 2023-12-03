package sample

import (
	"fmt"

	"github.com/buarki/supervised-machine-learning/matrix"
	"github.com/buarki/supervised-machine-learning/normalize"
)

type Sample struct {
	Input  *matrix.Matrix
	Output *matrix.Matrix
}

type WeightsSample struct {
	W2 *matrix.Matrix
	W3 *matrix.Matrix
}

func GetWeights() (*WeightsSample, error) {
	knownW2, err := matrix.New(2, 3, []float64{
		0.03676822608929586, 0.8260280470535342, -0.972356571577406,
		-0.043683543466855435, 0.8162020654008229, 0.8408781057976225,
	})
	if err != nil {
		return nil, err
	}
	knownW3, err := matrix.New(3, 1, []float64{
		0.4369289472888507,
		-0.7286028295881518,
		-0.7431793967666263,
	})
	if err != nil {
		return nil, err
	}
	return &WeightsSample{
		W2: knownW2,
		W3: knownW3,
	}, nil
}

type BiasSample struct {
	B2 *matrix.Matrix
	B3 *matrix.Matrix
}

func GetBiases() (*BiasSample, error) {
	knownB2, err := matrix.New(3, 3, []float64{
		0.8253202608929586, 0.7053534282602804, -0.8408781071577406,
		-0.927252738366855435, -0.6540088162020229, 0.97235650593976225,
		-0.92028292938366855435, -1.202022939292289, -0.3339999997762235,
	})
	if err != nil {
		return nil, fmt.Errorf("expected error to be nil, got %v", err)
	}
	knownB3, err := matrix.New(3, 1, []float64{
		-0.0729742481518,
		0.971939328807,
		-1.200800976263,
	})
	if err != nil {
		return nil, fmt.Errorf("expected error to be nil, got %v", err)
	}
	return &BiasSample{
		B2: knownB2,
		B3: knownB3,
	}, nil
}

func GetAReadyInputAndOutputSample() (*Sample, error) {
	rawX, err := createInput()
	if err != nil {
		return nil, err
	}
	normalizedX, err := normalize.Input(rawX)
	if err != nil {
		return nil, err
	}
	rawY, err := createOutput()
	if err != nil {
		return nil, err
	}
	normalizedY, err := normalize.Output(rawY)
	if err != nil {
		return nil, err
	}
	return &Sample{
		Input:  normalizedX,
		Output: normalizedY,
	}, nil
}

func createInput() (*matrix.Matrix, error) {
	data := []float64{3, 5, 5, 1, 10, 2}
	rows, cols := 3, 2
	matrix, err := matrix.New(rows, cols, data)
	if err != nil {
		return nil, err
	}
	return matrix, nil
}

func createOutput() (*matrix.Matrix, error) {
	data := []float64{75, 82, 93}
	rows, cols := 3, 1
	matrix, err := matrix.New(rows, cols, data)
	if err != nil {
		return nil, err
	}
	return matrix, nil
}
