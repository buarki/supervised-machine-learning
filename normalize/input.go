package normalize

import "github.com/buarki/supervised-machine-learning/matrix"

func Input(m *matrix.Matrix) (*matrix.Matrix, error) {
	rows, cols := m.Rows, m.Columns
	normalizedInput, err := matrix.New(rows, cols, m.FlattenedElements())
	if err != nil {
		return nil, err
	}
	for j := 0; j < cols; j++ {
		maxValue, err := normalizedInput.GetAt(0, j)
		if err != nil {
			return nil, err
		}
		// Find the maximum value in column j
		for i := 1; i < rows; i++ {
			maxColumnElement, err := normalizedInput.GetAt(i, j)
			if err != nil {
				return nil, err
			}
			if maxColumnElement > maxValue {
				maxValue = maxColumnElement
			}
		}
		// Normalize each element in column j
		for i := 0; i < rows; i++ {
			element, err := normalizedInput.GetAt(i, j)
			if err != nil {
				return nil, err
			}
			if err := normalizedInput.SetAt(i, j, element/maxValue); err != nil {
				return nil, err
			}
		}
	}
	return normalizedInput, nil
}
