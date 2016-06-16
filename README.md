# cyclades

# Implementation details (Work in progress)
## class DataPoint (interface)

Represents a single datapoint to be used in training.
### Methods
```c++
Datapoint(const string & input_line, int order)
- Initialize constructed datapoint from a single input line from the input data. Order represents
  the order in which the datapoint is to be processed.
```

```c++
const std::vector<double> GetLabels()
- Return labels for datapoint.
```

```c++
const std::vector<int> GetCoordinates()
- Get coordinates corresponding to labels of GetLabels().
```

```c++
int GetNumCoordinateTouches()
- Return the number of coordinates that this datapoint touches.
```

## class Model (interface)

Represents the model to train. Define any necessary extra variables to represent the model.
### Methods
```c++
Model(const string &input_line)
- Initialize the model from a single input line from the input data. This line should be the top line of the data file.
```

```c++
double ComputeLoss(const std::vector<Datapoint *> &datapoints)
- Computes loss given set of datapoints.
```

## class DatasetReader

Allocates and initializes datapoints and model from given input file. Lines from the data file
are fed into the Datapoint and Model objects for initialization.

### File format
- 1st line : Model configuration input line.
- 2nd line+ : Input lines for DataPoint.

### Methods
```c++
static void ReadDataset(string input_file, vector<Datapoint *> &datapoints, Model **model);
- Initialize/Allocates model and datapoints from input file.
```

## class Trainer (interface)

Interface for Hogwild/Cyclades trainers.
### Methods
```c++
void Train(Model *model, const std::vector<Datapoint *> & datapoints)
- Trains model on datapoints.
```

## class Partitioner (interface)

Partitions datapoints to workloads for multiple threads. Should have different types, like
BasicPartitioner, CycladesPartitioner, etc.
### Methods
```c++
DatapointPartitions Partition(const std::vector<Datapoint *> &datapoints, int n_threads)
- Main partitioning method, which, given datapoints and number of threads,
  returns a DatapointPartitions object representing partitioned datapoints for number of threads.
```

## class DatapointPartitions

Represents partitions of datapoints for a given number of threads.
Batches are a way to group datapoints, necessary for Cyclades partitioning.
### Methods
```c++
DatapointPartitions(int n_threads)
- Constructor which takes in n_threads to represent partitions for.
```

```c++
void StartNewBatch()
- Starts a new batch.
```

```c++
int NumBatches()
- Returns number of batches in the partition.
```

```c++
int NumDatapointsInBatch(int thread, int batch)
- Returns the number of datapoints for a given thread in a given batch.
```

```c++
void AddDatapointToThread(Datapoint *datapoint, int thread)
- Adds a datapoint to the current batch for the given thread.
```
