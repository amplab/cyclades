# cyclades

# Implementation

## Design details
### class: DataPoint (potentially override with application specific implementation)
#### Description
Represents a single datapoint to be used in training.
#### Override functions.
void Initialize(string input_line)
- Initialize constructed datapoint from a single input line from the input data.

### class: DatasetReader
#### Description
Creates a vector of DataPoints given an input data file.
#### Necessary functions
static ReadDataset(string input_file, vector<DataPoint> &datapoints);
- Initialize datapoints from input file.
