package matrix

import (
	"fmt"
	"math"
	"strings"
)

const (
	ElementWiseOperationSum            = "+"
	ElementWiseOperationSubtraction    = "-"
	ElementWiseOperationMultiplication = "*"
	ElementWiseOperationDivision       = "/"
)

var (
	scalarOperation = map[string]func(a, b float64) float64{
		ElementWiseOperationSum: func(a, b float64) float64 {
			return a + b
		},
		ElementWiseOperationSubtraction: func(a, b float64) float64 {
			return a - b
		},
		ElementWiseOperationMultiplication: func(a, b float64) float64 {
			return a * b
		},
		ElementWiseOperationDivision: func(a, b float64) float64 {
			return a / b
		},
	}
)

type Matrix struct {
	Rows    int
	Columns int
	data    [][]float64
}

func New(rows, columns int, flattenData []float64) (*Matrix, error) {
	matrix, err := emptyMatrix(rows, columns)
	if err != nil {
		return nil, err
	}
	if len(flattenData) != (rows * columns) {
		return nil, fmt.Errorf("provided array cannot be arranged on a matrix of size %dx%d", rows, columns)
	}
	k := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			matrix.data[i][j] = flattenData[k]
			k++
		}
	}
	return matrix, nil
}

func emptyMatrix(rows, columns int) (*Matrix, error) {
	if rows <= 0 {
		return nil, fmt.Errorf("rows param must be > 0, received %v", rows)
	}
	if columns <= 0 {
		return nil, fmt.Errorf("columns param must be > 0, received %v", rows)
	}
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, columns)
	}
	return &Matrix{
		Rows:    rows,
		Columns: columns,
		data:    data,
	}, nil
}

// FlattenedElements returns all matrix elements on a slice
func (m *Matrix) FlattenedElements() []float64 {
	flattenedElements := make([]float64, m.Rows*m.Columns)
	index := 0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			flattenedElements[index] = m.data[i][j]
			index++
		}
	}
	return flattenedElements
}

// SumWith sums placehoder matrix with the given one
// and returns the sum matrix if the sum can be done.
func (m *Matrix) SumWith(a *Matrix) (*Matrix, error) {
	return m.elementWiseOperation(a, ElementWiseOperationSum)
}

// Minus perfoms the subtraction from placeholder's elements
// and retuns the subtraction matrixi if the operation can be done.
func (m *Matrix) Minus(a *Matrix) (*Matrix, error) {
	return m.elementWiseOperation(a, ElementWiseOperationSubtraction)
}

// DotProductWith perfoms the dot product between two matrices
// and retuns the result matrix if the operation can be done.
//
// Note: bellow implementation was designed for small matrices as it is O(n^3)
func (m *Matrix) DotProductWith(a *Matrix) (*Matrix, error) {
	if a == nil {
		return nil, fmt.Errorf("given matrix is nil")
	}
	dotProductCannotBeDone := !(m.Columns == a.Rows)
	if dotProductCannotBeDone {
		return nil, fmt.Errorf("dot product not possible due to matrix dimensions, this has shape (%dx%d), given has (%dx%d)", m.Rows, m.Columns, a.Rows, a.Columns)
	}
	dotProductMatrix, err := emptyMatrix(m.Rows, a.Columns)
	if err != nil {
		return nil, err
	}
	for i := range dotProductMatrix.data {
		dotProductMatrix.data[i] = make([]float64, a.Columns)
		for j := 0; j < a.Columns; j++ {
			for k := 0; k < m.Columns; k++ {
				dotProductMatrix.data[i][j] += m.data[i][k] * a.data[k][j]
			}
		}
	}
	return dotProductMatrix, nil
}

// HadamardProduct executes the Hadamard product and
// return the result matrix if the operation is possible.
func (m *Matrix) HadamardProductWith(a *Matrix) (*Matrix, error) {
	return m.elementWiseOperation(a, ElementWiseOperationMultiplication)
}

// SumOfAllElements sums all elements present and
// return the sum.
func (m *Matrix) SumOfAllElements() float64 {
	sum := 0.0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			sum += m.data[i][j]
		}
	}
	return sum
}

// T returns the matrix transpose
func (m *Matrix) T() *Matrix {
	transposedMatrix := &Matrix{
		Rows:    m.Columns,
		Columns: m.Rows,
		data:    make([][]float64, m.Columns),
	}
	for i := range transposedMatrix.data {
		transposedMatrix.data[i] = make([]float64, m.Rows)
		for j := range transposedMatrix.data[i] {
			transposedMatrix.data[i][j] = m.data[j][i]
		}
	}
	return transposedMatrix
}

func (m *Matrix) elementWiseOperation(a *Matrix, operation string) (*Matrix, error) {
	if a == nil {
		return nil, fmt.Errorf("given matrix is nil")
	}
	elementWiseOperationCannotBeDone := !(m.Rows == a.Rows && m.Columns == a.Columns)
	if elementWiseOperationCannotBeDone {
		return nil, fmt.Errorf("matrices have different dimensions: this has (%d x %d) while given one has (%d x %d)", m.Rows, m.Columns, a.Rows, a.Columns)
	}
	sumMatrix, err := emptyMatrix(m.Rows, m.Columns)
	if err != nil {
		return nil, err
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			mValue, err := m.GetAt(i, j)
			if err != nil {
				return nil, err
			}
			aValue, err := a.GetAt(i, j)
			if err != nil {
				return nil, err
			}
			if err := sumMatrix.SetAt(i, j, scalarOperation[operation](mValue, aValue)); err != nil {
				return nil, err
			}
		}
	}
	return sumMatrix, nil
}

func (m *Matrix) SetAt(rowIndex, columnIndex int, value float64) error {
	if err := m.checkBounds(rowIndex, columnIndex); err != nil {
		return err
	}
	m.data[rowIndex][columnIndex] = value
	return nil
}

func (m *Matrix) GetAt(rowIndex, columnIndex int) (float64, error) {
	if err := m.checkBounds(rowIndex, columnIndex); err != nil {
		return 0, err
	}
	return m.data[rowIndex][columnIndex], nil
}

// ApplyElementWise takes a function as argument, applies it
// element wise and returns a new matrix with the function applied.
func (m *Matrix) ApplyElementWise(f func(value float64) float64) (*Matrix, error) {
	if f == nil {
		return nil, fmt.Errorf("function to be applied element wise must be passed")
	}
	newMatrix, err := emptyMatrix(m.Rows, m.Columns)
	if err != nil {
		return nil, err
	}
	for i := 0; i < newMatrix.Rows; i++ {
		for j := 0; j < newMatrix.Columns; j++ {
			originalMatrixElement, err := m.GetAt(i, j)
			if err != nil {
				return nil, err
			}
			if err := newMatrix.SetAt(i, j, f(originalMatrixElement)); err != nil {
				return nil, err
			}
		}
	}
	return newMatrix, nil
}

// Norm implements the norm of a matrix based on argument norm.
// norm = 1 means L1 norm, norm = 2 means L2 norm
func (m *Matrix) Norm(norm int) (float64, error) {
	if norm != 1 && norm != 2 {
		return 0, fmt.Errorf("unsupported norm, only L1 (norm=1) and L2 (norm=2) are supported")
	}
	var result float64
	switch norm {
	case 1:
		// L1 Norm (sum of absolute values of elements)
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Columns; j++ {
				result += math.Abs(m.data[i][j])
			}
		}
	default:
		// L2 Norm (square root of the sum of squared values)
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Columns; j++ {
				result += m.data[i][j] * m.data[i][j]
			}
		}
		result = math.Sqrt(result)
	}
	return result, nil
}

func (m *Matrix) checkBounds(rowIndex, columnIndex int) error {
	if rowIndex >= m.Rows {
		return fmt.Errorf("rowIndex out of bounds, matrix has rows [%d-%d]", 0, len(m.data)-1)
	}
	if columnIndex >= m.Columns {
		return fmt.Errorf("columnIndex out of bounds, matrix has columns [%d-%d]", 0, len(m.data[0])-1)
	}
	if rowIndex < 0 {
		return fmt.Errorf("row index must be >= 0, received %v", rowIndex)
	}
	if columnIndex < 0 {
		return fmt.Errorf("column index must be >= 0, received %v", rowIndex)
	}
	return nil
}

func (m *Matrix) ToString() string {
	var sb strings.Builder
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			sb.WriteString(fmt.Sprintf("%v", m.data[i][j]))
			if j < m.Columns-1 {
				sb.WriteString(" ")
			}
		}
		if i < m.Rows-1 {
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

// FrobeniusNormRatio is an application of Frobenius norm. It performs the
// (L2 norm of difference) / (L2 norm of sum) and return a scalar with the result.
// The result indicates how similar those matrix are. The smaller the output is more
// similar they are, and the value 1e-8 can be used as a valid offset of similarity.
func (m *Matrix) FrobeniusNormRatio(a *Matrix) (float64, error) {
	if m.Rows != a.Rows || m.Columns != a.Columns {
		return 0, fmt.Errorf("given matrix has not the same dimensions, placeholder is (%dx%d), given is (%dx%d)", m.Rows, m.Columns, a.Rows, a.Columns)
	}
	w2Diff, err := m.Minus(a)
	if err != nil {
		return 0, fmt.Errorf("failed to compute placehoder - given matrix, got %v", err)
	}
	diffNorm, err := w2Diff.Norm(2)
	if err != nil {
		return 0, fmt.Errorf("failed to compute the norm of difference matrix, got %v", err)
	}
	w2Sum, err := m.SumWith(a)
	if err != nil {
		return 0, fmt.Errorf("failed to compute placehoder + given matrix, got %v", err)
	}
	sumNorm, err := w2Sum.Norm(2)
	if err != nil {
		return 0, fmt.Errorf("failed to compute the norm of sum matrix, got %v", err)
	}
	return diffNorm / sumNorm, nil
}
