package normalize

import "github.com/buarki/supervised-machine-learning/matrix"

const (
	MaxTestScore = 10.0
)

func Output(m *matrix.Matrix) (*matrix.Matrix, error) {
	normalizedMatrix, err := matrix.New(m.Rows, m.Columns, m.FlattenedElements())
	if err != nil {
		return nil, err
	}
	normalized, err := normalizedMatrix.ApplyElementWise(func(value float64) float64 {
		return value / MaxTestScore
	})
	if err != nil {
		return nil, err
	}
	return normalized, nil
}
