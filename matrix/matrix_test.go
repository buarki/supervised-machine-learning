package matrix_test

import (
	"math"
	"strings"
	"testing"

	"github.com/buarki/supervised-machine-learning/matrix"
)

func TestCretingAMatrixWithInvalidRowValue(t *testing.T) {
	invalidRowValue := -1
	validColumn := 1
	m, err := matrix.New(invalidRowValue, validColumn, []float64{})
	if m != nil {
		t.Errorf("expected m to be nill")
	}
	if err == nil {
		t.Errorf("expected err to be not nil")
	}
}

func TestCretingAMatrixWithInvalidColumValue(t *testing.T) {
	invalidRowValue := 1
	validColumn := -1
	m, err := matrix.New(invalidRowValue, validColumn, []float64{})
	if m != nil {
		t.Errorf("expected m to be nill")
	}
	if err == nil {
		t.Errorf("expected err to be not nil")
	}
}

func TestCretingAMatrixWithDimensionsParamsNotSyncWithFlattenArray(t *testing.T) {
	invalidRowValue := 1
	validColumn := 2
	flattenArray := []float64{1, 2, 3}
	m, err := matrix.New(invalidRowValue, validColumn, flattenArray)
	if err == nil {
		t.Errorf("expected err to be not nil")
	}
	if m != nil {
		t.Errorf("expected m to be nil")
	}
}

func TestCretingAMatrixWithDimensionsParamsNotSyncWithFlattenArray1(t *testing.T) {
	invalidRowValue := 1
	validColumn := 2
	flattenArray := []float64{1}
	m, err := matrix.New(invalidRowValue, validColumn, flattenArray)
	if err == nil {
		t.Errorf("expected err to be not nil")
	}
	if m != nil {
		t.Errorf("expected m to be nil")
	}
}

func TestCretingAValidArray(t *testing.T) {
	validRow := 1
	validColumn := 2
	flattenArray := []float64{1, 2}
	m, err := matrix.New(validRow, validColumn, flattenArray)
	if err != nil {
		t.Errorf("expected err to be nil")
	}
	if m == nil {
		t.Errorf("expected m to NOT nill")
	}
}

func TestToStringMethod(t *testing.T) {
	validRows := 2
	validColumns := 2
	flattenArray := []float64{1, 2, 3, 4}

	m, err := matrix.New(validRows, validColumns, flattenArray)
	asString := m.ToString()

	if err != nil {
		t.Errorf("expected err to be nil")
	}
	if m == nil {
		t.Errorf("expected m to NOT nill")
	}
	if !strings.Contains(asString, "1") {
		t.Errorf("expected matrix string to have 1")
	}
	if !strings.Contains(asString, "2") {
		t.Errorf("expected matrix string to have 2")
	}
	if !strings.Contains(asString, "3") {
		t.Errorf("expected matrix string to have 3")
	}
	if !strings.Contains(asString, "4") {
		t.Errorf("expected matrix string to have 4")
	}
}

func TestGetWithRowIndexLessThan0(t *testing.T) {
	validRows := 2
	validColumns := 2
	flattenArray := []float64{1, 2, 3, 4}
	invalidRowIndex := -1

	m, err := matrix.New(validRows, validColumns, flattenArray)

	if err != nil {
		t.Errorf("expected err to be nil")
	}

	_, err = m.GetAt(invalidRowIndex, 0)
	if err == nil {
		t.Errorf("expected err to be not nil")
	}
}

func TestGetWithColumnIndexLessThan0(t *testing.T) {
	validRows := 2
	validColumns := 2
	flattenArray := []float64{1, 2, 3, 4}
	invalidColumnIndex := -1

	m, err := matrix.New(validRows, validColumns, flattenArray)

	if err != nil {
		t.Errorf("expected err to be nil")
	}

	_, err = m.GetAt(0, invalidColumnIndex)
	if err == nil {
		t.Errorf("expected err to be not nil")
	}
}

func TestGetWithColumnIndexBiggerThanColumns(t *testing.T) {
	validRows := 2
	validColumns := 2
	flattenArray := []float64{1, 2, 3, 4}
	invalidColumnIndex := 3

	m, err := matrix.New(validRows, validColumns, flattenArray)

	if err != nil {
		t.Errorf("expected err to be nil")
	}

	_, err = m.GetAt(0, invalidColumnIndex)
	if err == nil {
		t.Errorf("expected err to be not nil")
	}
}

func TestGetWithRowIndexBiggerThanRows(t *testing.T) {
	validRows := 2
	validColumns := 2
	flattenArray := []float64{1, 2, 3, 4}
	invalidRowIndex := 3

	m, err := matrix.New(validRows, validColumns, flattenArray)

	if err != nil {
		t.Errorf("expected err to be nil")
	}

	_, err = m.GetAt(invalidRowIndex, 0)
	if err == nil {
		t.Errorf("expected err to be not nil")
	}
}

func TestGetItemAtValidPosition(t *testing.T) {
	validRows := 2
	validColumns := 2
	flattenArray := []float64{1, 2, 3, 4}
	validRowIndex := 0
	validColumnIndex := 0

	m, err := matrix.New(validRows, validColumns, flattenArray)

	if err != nil {
		t.Errorf("expected err to be nil")
	}

	value, err := m.GetAt(validRowIndex, validColumnIndex)
	if err != nil {
		t.Errorf("expected err to be nil")
	}
	if value != flattenArray[0] {
		t.Errorf("expected element to be 1, got %v", value)
	}
}

func TestSetItemAtInvalidPosition(t *testing.T) {
	validRows := 2
	validColumns := 2
	flattenArray := []float64{1, 2, 3, 4}
	invalidRow := 2
	validColumnIndex := 0

	m, err := matrix.New(validRows, validColumns, flattenArray)

	if err != nil {
		t.Errorf("expected err to be nil")
	}

	err = m.SetAt(invalidRow, validColumnIndex, 203)
	if err == nil {
		t.Errorf("expected err to be nil")
	}
}

func TestSetItemAtValidPosition(t *testing.T) {
	validRows := 2
	validColumns := 2
	flattenArray := []float64{1, 2, 3, 4}
	validRow := 1
	validColumnIndex := 0

	m, err := matrix.New(validRows, validColumns, flattenArray)

	if err != nil {
		t.Errorf("expected err to be nil")
	}

	err = m.SetAt(validRow, validColumnIndex, 203)
	if err != nil {
		t.Errorf("expected err to be NOT nil")
	}
}

func TestSumTwoMatricesWithDifferentDimensions(t *testing.T) {
	m1, err := matrix.New(1, 1, []float64{1})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	m2, err := matrix.New(1, 2, []float64{1, 32})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	result, err := m1.SumWith(m2)
	if err == nil {
		t.Errorf("expected err to be  not nil")
	}
	if result != nil {
		t.Errorf("expected result to be nil")
	}
}

func TestSumTwoMatricesWithBeingNil(t *testing.T) {
	m1, err := matrix.New(1, 1, []float64{1})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	var m2 *matrix.Matrix

	result, err := m1.SumWith(m2)
	if err == nil {
		t.Errorf("expected err to be  not nil")
	}
	if result != nil {
		t.Errorf("expected result to be nil")
	}
}

func TestSumTwoMatricesWithSameDimensions(t *testing.T) {
	m1, err := matrix.New(1, 1, []float64{1})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	m2, err := matrix.New(1, 1, []float64{2})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	result, err := m1.SumWith(m2)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	if result == nil {
		t.Errorf("expected result to be NOT nil")
	}
	if result.Rows != m1.Rows {
		t.Errorf("expected rows to be [%v], got [%v]", m1.Rows, result.Rows)
	}
	if result.Columns != m1.Columns {
		t.Errorf("expected columns to be [%v], got [%v]", m1.Columns, result.Columns)
	}

	value, err := result.GetAt(0, 0)
	if err != nil {
		t.Errorf("expected err to be nil")
	}

	if value != 3 {
		t.Errorf("expected value to be 3, got %v", value)
	}
}

func TestSubstractTwoMatrices(t *testing.T) {
	m1, err := matrix.New(1, 1, []float64{1})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	m2, err := matrix.New(1, 1, []float64{2})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	result, err := m1.Minus(m2)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	if result == nil {
		t.Errorf("expected result to be NOT nil")
	}
	if result.Rows != m1.Rows {
		t.Errorf("expected rows to be [%v], got [%v]", m1.Rows, result.Rows)
	}
	if result.Columns != m1.Columns {
		t.Errorf("expected columns to be [%v], got [%v]", m1.Columns, result.Columns)
	}

	value, err := result.GetAt(0, 0)
	if err != nil {
		t.Errorf("expected err to be nil")
	}

	if value != -1 {
		t.Errorf("expected value to be 3, got %v", value)
	}
}

func TestMultiplicationNotPossibleDueToDimensions(t *testing.T) {
	m1, err := matrix.New(1, 1, []float64{1})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	m2, err := matrix.New(2, 1, []float64{2, 2})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	result, err := m1.DotProductWith(m2)
	if err == nil {
		t.Errorf("expected error to be not nil")
	}
	if result != nil {
		t.Errorf("expected result to be nil, got %v", result)
	}
}

func TestMultiplicationPossible(t *testing.T) {
	m1, err := matrix.New(2, 1, []float64{2, 2})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	m2, err := matrix.New(1, 1, []float64{1})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	result, err := m1.DotProductWith(m2)

	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if result == nil {
		t.Errorf("expected result to be not nil")
	}
	if result.Rows != m1.Rows {
		t.Errorf("expected result matrix has rows %d, got %d", m1.Rows, result.Rows)
	}
	if result.Columns != m1.Columns {
		t.Errorf("expected result matrix has Columns %d, got %d", m1.Columns, result.Columns)
	}
}

func TestTranspose(t *testing.T) {
	m1, err := matrix.New(2, 1, []float64{2, 2})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	m1T := m1.T()

	if m1T.Rows != m1.Columns {
		t.Errorf("expected transpose to %d rows, got %d", m1.Columns, m1T.Rows)
	}
	if m1T.Columns != m1.Rows {
		t.Errorf("expected transpose to %d columns, got %d", m1.Rows, m1T.Columns)
	}
	element00T, err := m1T.GetAt(0, 0)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	element00, err := m1.GetAt(0, 0)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if element00 != element00T {
		t.Errorf("expected element 00 of transpose to be equal to the 00 on original, got %v", element00T)
	}
}

func TestTestHadamardProduct(t *testing.T) {
	m1, err := matrix.New(2, 2, []float64{1, 2, 3, 4})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	m2, err := matrix.New(2, 2, []float64{5, 6, 7, 8})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	result, err := m1.HadamardProductWith(m2)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	result00, _ := result.GetAt(0, 0)
	if result00 != 5 {
		t.Errorf("expected element 00 to be 3, got %v", result00)
	}
	result01, _ := result.GetAt(0, 1)
	if result01 != 12 {
		t.Errorf("expected element 01 to be 12, got %v", result01)
	}
	result10, _ := result.GetAt(1, 0)
	if result10 != 21 {
		t.Errorf("expected element 10 to be 21, got %v", result10)
	}
	result11, _ := result.GetAt(1, 1)
	if result11 != 32 {
		t.Errorf("expected element 11 to be 32, got %v", result11)
	}
}

func TestSum(t *testing.T) {
	numbers := []float64{
		-11, 17, 3,
		5, 33, 7,
		8.98, -9, 2.3,
	}
	expectedSum := 56.28
	m1, err := matrix.New(3, 3, numbers)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	sum := m1.SumOfAllElements()

	if sum != expectedSum {
		t.Errorf("expected sum to be %v, got %v", expectedSum, sum)
	}
}

func TestApplyElementWiseWithNilFunction(t *testing.T) {
	m1, err := matrix.New(2, 2, []float64{1, 2, 3, 4})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	applied, err := m1.ApplyElementWise(nil)
	if err == nil {
		t.Errorf("expected error to be not nil")
	}
	if applied != nil {
		t.Errorf("expected applie to be nil, got %v", applied)
	}
}

func TestApplyElementWise(t *testing.T) {
	m1, err := matrix.New(2, 2, []float64{1, 2, 3, 4})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	functionToDoubleValues := func(value float64) float64 {
		return value * 2.0
	}

	applied, err := m1.ApplyElementWise(functionToDoubleValues)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	result00, _ := applied.GetAt(0, 0)
	if result00 != 2 {
		t.Errorf("expected element 00 to be 2, got %v", result00)
	}
	result01, _ := applied.GetAt(0, 1)
	if result01 != 4 {
		t.Errorf("expected element 01 to be 4, got %v", result01)
	}
	result10, _ := applied.GetAt(1, 0)
	if result10 != 6 {
		t.Errorf("expected element 10 to be 6, got %v", result10)
	}
	result11, _ := applied.GetAt(1, 1)
	if result11 != 8 {
		t.Errorf("expected element 11 to be 8, got %v", result11)
	}
}

func TestNormL1(t *testing.T) {
	m1, err := matrix.New(2, 2, []float64{7, -3, 11, 9.3})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	expectedL1Norm := 30.3

	norm, err := m1.Norm(1)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	if norm != expectedL1Norm {
		t.Errorf("expected norm to be %v, got %v", expectedL1Norm, norm)
	}
}

func TestNormL2(t *testing.T) {
	m1, err := matrix.New(2, 2, []float64{7, -3, 11, 9.3})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	expectedL2Norm := 16.29
	delta := 1e-8

	norm, err := m1.Norm(2)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	diff := math.Abs(norm - expectedL2Norm)
	if diff < delta {
		t.Errorf("expected diff to be %v, got %v", delta, diff)
	}
}

func TestNormWithInvalidParam(t *testing.T) {
	m1, err := matrix.New(2, 2, []float64{7, -3, 11, 9.3})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	_, err = m1.Norm(3)
	if err == nil {
		t.Errorf("expected err to be nil")
	}
}

func TestFlattenedElements(t *testing.T) {
	m1, err := matrix.New(2, 2, []float64{7, -3, 11, 9.3})
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	flattened := m1.FlattenedElements()

	if len(flattened) != m1.Rows*m1.Columns {
		t.Errorf("expected flattened slice to have length %d, got %d", m1.Rows*m1.Columns, len(flattened))
	}
}

func TestFrobeniusNormRatioForEqualMatrices(t *testing.T) {
	data := []float64{7, -3, 11, 9.3}
	A, err := matrix.New(2, 2, data)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	B, err := matrix.New(2, 2, data)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	diff, err := A.FrobeniusNormRatio(B)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	if diff != 0 {
		t.Errorf("expected diff to be zero, got %v", err)
	}
}
