package normalize_test

import (
	"testing"

	"github.com/buarki/supervised-machine-learning/matrix"
	"github.com/buarki/supervised-machine-learning/normalize"
)

func TestInputNormalization(t *testing.T) {
	data := []float64{3.0, 5.0, 5.0, 1.0, 10.0, 2.0}
	maxValueOnColumn1 := 10.0
	maxValueOnColumn2 := 5.0
	x, err := matrix.New(3, 2, data)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	normalizedX, err := normalize.Input(x)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	// Checking column 1
	e00, err := normalizedX.GetAt(0, 0)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e00 != data[0]/maxValueOnColumn1 {
		t.Errorf("expected element 00 to be [%v], got [%v]", data[0]/maxValueOnColumn1, e00)
	}
	e10, err := normalizedX.GetAt(1, 0)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e10 != data[2]/maxValueOnColumn1 {
		t.Errorf("expected element 10 to be [%v], got [%v]", data[2]/maxValueOnColumn1, e10)
	}
	e20, err := normalizedX.GetAt(2, 0)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e20 != data[4]/maxValueOnColumn1 {
		t.Errorf("expected element 20 to be [%v], got [%v]", data[4]/maxValueOnColumn1, e20)
	}

	// Checking column 2
	e01, err := normalizedX.GetAt(0, 1)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e01 != data[1]/maxValueOnColumn2 {
		t.Errorf("expected element 01 to be [%v], got [%v]", data[1]/maxValueOnColumn2, e01)
	}
	e11, err := normalizedX.GetAt(1, 1)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e11 != data[3]/maxValueOnColumn2 {
		t.Errorf("expected element 11 to be [%v], got [%v]", data[3]/maxValueOnColumn2, e11)
	}
	e12, err := normalizedX.GetAt(2, 1)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e12 != data[5]/maxValueOnColumn2 {
		t.Errorf("expected element 12 to be [%v], got [%v]", data[5]/maxValueOnColumn2, e12)
	}
}
