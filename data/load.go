package data

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

type Sample struct {
	HoursOfSleep      int
	HoursOfMeditation int
	ScoreTest         int
}

// LoadDataSetFrom opens a CSV file from a given path
// and returns a list of samples
func LoadDataSetFrom(path string) ([]Sample, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("error opening CSV file from path %s, got %v", path, err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error reading CSV file from path %s, got %v", path, err)
	}
	var sample []Sample
	for i, record := range records {
		if i == 0 {
			continue
		}
		hoursOfSleep, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing hours of sleep: %v", err)
		}
		hoursOfMeditation, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing hours of meditation: %v", err)
		}
		testScore, err := strconv.ParseFloat(record[2], 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing test score: %v", err)
		}
		sample = append(sample, Sample{HoursOfSleep: int(hoursOfSleep), HoursOfMeditation: int(hoursOfMeditation), ScoreTest: int(testScore)})
	}
	return sample, nil
}
