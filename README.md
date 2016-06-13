# cyclades

## Design details (Work in progress)
### class: DataPoint (override)
#### Description
Represents a single datapoint to be used in training.
#### Override functions
void Initialize(string input_line)
- Initialize constructed datapoint from a single input line from the input data.

### class: Model (override)
#### Description
Represents the model to train. Define any necessary extra variables to represent the model.
#### Override functions
void Initialize(string input_line)
- Initialize the model from a single input line from the input data. This line should be the top line of the data file.

### class: DatasetReader
#### Description
Initializes datapoints and model from given input file.
#### Necessary functions
static ReadDataset(string input_file, vector<DataPoint> &datapoints, Model &model);
- Initialize datapoints from input file.
#### File format
- 1st line : Model configuration input line.
- 2nd line : Number of datapoints.
- 3rd line+ : Input lines for DataPoint.
