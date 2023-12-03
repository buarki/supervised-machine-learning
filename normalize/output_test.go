package normalize_test

import (
	"testing"

	"github.com/buarki/supervised-machine-learning/matrix"
	"github.com/buarki/supervised-machine-learning/normalize"
)

func TestOutputNormalization(t *testing.T) {
	data := []float64{100, 200, 300, 400}
	m1, err := matrix.New(2, 2, data)
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}

	normalizedM1, err := normalize.Output(m1)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}

	e00, err := normalizedM1.GetAt(0, 0)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e00 != data[0]/normalize.MaxTestScore {
		t.Errorf("expected element 00 to be [%v], got [%v]", data[0]/normalize.MaxTestScore, e00)
	}
	e01, err := normalizedM1.GetAt(0, 1)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e01 != data[1]/normalize.MaxTestScore {
		t.Errorf("expected element 01 to be [%v], got [%v]", data[1]/normalize.MaxTestScore, e01)
	}
	e10, err := normalizedM1.GetAt(1, 0)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e10 != data[2]/normalize.MaxTestScore {
		t.Errorf("expected element 10 to be [%v], got [%v]", data[2]/normalize.MaxTestScore, e10)
	}
	e11, err := normalizedM1.GetAt(1, 1)
	if err != nil {
		t.Errorf("expected error to be nil, got %v", err)
	}
	if e11 != data[3]/normalize.MaxTestScore {
		t.Errorf("expected element 11 to be [%v], got [%v]", data[3]/normalize.MaxTestScore, e11)
	}
}
