# inatVisionTraining

## Prerequisites

The following python modules are required:

* numpy
* pandas
* yaml
* tensorflow
* tqdm
* json

## Configuration

Specify various parameters for training, validating, and testing the model in a YAML file. An example is provided in `config.yml.example`, though the parameters within are not up to date.

The following tables describe parameters that **must be set**. There are no defaults within the code.

### Storage Parameters

Parameters for the storage of log files, checkpoints, etc.

Name                | Function
--------------------|----------
TENSORBOARD_LOG_DIR | Directory for saving log information
CHECKPOINT_DIR      | Directory for storing checkpoints
FINAL_SAVE_DIR      | Directory for saving the final training data
BACKUP_DIR          | Directory for storing backup data; this is an experimental Keras feature

### Data Parameters

Parameters for specifying dataset location and metadata. Datasets should be stored in JSON format and are read in using [Pandas](https://pandas.pydata.org/).

Name              | Function
------------------|----------
TRAINING_DATA     | JSON file with training data
VAL_DATA          | JSON file with validation data
TEST_DATA         | JSON file with test data
NUM_CLASSES       | Number of output classes
LABEL_COLUMN_NAME | Name of the column containing class labels

### Training Parameters

Parameters for training the model. The training framework is hardcoded to use the "rmsprop" optimizer.

Name                  | Function
----------------------|----------
TRAIN_MIXED_PRECISION | Set to "True" only if your GPU supports CUDA >= 7.0
MULTIGPU              | Boolean; use more than one GPU
BATCH_SIZE            | Size of batch per GPU
NUM_EPOCHS            | Number of training epochs
INITIAL_LEARNING_RATE | Initial learning rate
LR_DECAY_FACTOR       | Learning rate decay factor
EPOCHS_PER_LR_DECAY   | Number of epochs over which to decay by 
RMSPROP_RHO           | Parameter "rho" for the "rmsprop" optimizer
RMSPROP_MOMENTUM      | Parameter "momentum" for the "rmsprop" optimizer
RMSPROP_EPSILON       | Parameter "epsilon" for the "rmsprop" optimizer

### Model Parameters

Parameters specifying the neural network.

Name             | Function
-----------------|----------
MODEL_NAME       | String to name the model
IMAGE_SIZE       | 2D list of integers specifying the size of each input image
DROPOUT_PCT      | Dropout percentage for the layer between "pool" and "logits"
PRETRAINED_MODEL | Path to existing model; cannot be named "imagenet"
DO_LABEL_SMOOTH   | Enable label smoothing in the categorical cross entropy loss function
LABEL_SMOOTH_MODE | Distribution The only currently supported mode is "flat"
LABEL_SMOOTH_PCT  | Percentage by which to smooth the labels

## Training and Evaluating the Model

### Training

After constructing a configuration file, the model can be trained by running `python train.py --config_file <config.yml>`.

### Model Evaluation

A trained model can be evaluated using `python eval.py --config_file <config.yml>`. This is the one required argument.

There are also the following optional arguments:

Argument           | Function
-------------------|----------
`--use_checkpoint` | If present, will evaluate using a checkpoint rather than the final export
`--should_save_results` | If yes, will save evaluation results to `--save_file`
`--save_file`           | File in which to save evaluation results. Will be `.npz`.