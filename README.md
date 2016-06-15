# cyclades

## Implementation details (Work in progress)
### class: DataPoint (interface)
#### Description
Represents a single datapoint to be used in training.
#### Override functions
void Initialize(const string & input_line)
- Initialize constructed datapoint from a single input line from the input data.

double GetLabel()
- Return label for datapoint.

void * GetData()
- Return pointer containing data representing datapoint. Should be precomputed for efficiency.

int GetNumCoordinateTouches
- Return the number of coordinates that this datapoint touches.

### class: Model (interface)
#### Description
Represents the model to train. Define any necessary extra variables to represent the model.
#### Override functions
void Initialize(const string &input_line)
- Initialize the model from a single input line from the input data. This line should be the top line of the data file.

double ComputeLoss(const std::vector<Datapoint *> &datapoints)
- Computes loss given set of datapoints.

### class: DatasetReader
#### Description
Allocates and initializes datapoints and model from given input file.
#### Methods
static void ReadDataset(string input_file, vector<Datapoint *> &datapoints, Model **model);
- Initialize/Allocates model and datapoints from input file.

#### File format
- 1st line : Model configuration input line.
- 2nd line : Number of datapoints.
- 3rd line+ : Input lines for DataPoint.

### class: Trainer (interface)
#### Description
Interface for Hogwild/Cyclades trainers.
#### Functions
void Run(Model *model, const std::vector<Datapoint *> & datapoints)
- Trains model on datapoints.