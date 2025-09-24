# Goal-Aware Neural SAT Solver

This repository contains some minimal working modifications to the official TensorFlow implementation of the following paper:
> **Goal-Aware Neural SAT Solver**
> 
> by  Emils Ozolins, Karlis Freivalds, Andis Draguns, Eliza Gaile, Ronalds Zakovskis, Sergejs Kozlovics 
> 
> Published in [2022 International Joint Conference on Neural Networks](https://ieeexplore.ieee.org/document/9892733)
>
> Also available in [arXiv](https://arxiv.org/abs/2106.07162)


## Requirements

To install requirements, run:

```sh
pip install -r requirements.txt
```

## Training

To train, run this command:
```sh
python3 -u main.py --train --model querysat --task <task_name>
```
This command will generate data in `/host-dir/data` for the task and train the selected model. 
The model checkpoints are saved in `/host-dir/querysat/<timestamp>`, you can pass `--label <name>` to the run command add a name to the checkpoints directory.
You can always change both directories in the `config.py` file.

If you have already populated the data directory with the training data, you can skip the data generation step by passing `--skip_data_gen` flag to the command.
By default, models are trained with 32 recurrent steps and evaluated with 64 steps. If you want to use other step counts for training or evaluation, please, change
`train_rounds` and `test_rounds` respectively in the model file. 

Valid task names are:
* `ksat` - for the k-SAT task with 3 to 100 variables;
* `3sat` - for the 3-SAT task consisting of hard formulas with 5 to 100 variables;
* `kcolor` - for the k-Color task for graphs with 4 to 40 vertices;
* `clique` - for the 3-Clique task for graphs with 4 to 40 vertices;
* `sha2019` - for the SHA-1 preimage attack from the [SAT Race 2019](http://sat-race-2019.ciirc.cvut.cz/) with 2-20 message bits.

A more detailed description of tasks and models are given in the [paper](https://arxiv.org/abs/2106.07162).
If you want to tweak any other aspects, please, modify `config.py`, the appropriate model file in `models\` or data file in `data\`.

## Evaluation

The trained model from the checkpoint can be evaluated on test set as follows:

```sh
python3 -u main.py --evaluate --task <task_name> --model querysat --restore <checkpoint_directory> --test_steps <test_steps>
```
By default, test sets for training and evaluating is generated with the same variable count. If you want to evaluate on larger formulas
please change `min_vars` and `max_vars` in generator code in suitable data generator `/data/`.

The `--test_steps` argument specifies the number of recurrent steps to use for evaluation. By default, it is set to 512.
As in the training case the test data will be generated if not found in the data directory. For evaluation on the benchmark dataset make sure the benchmark data is formatted in DIMACS format and placed in the appropriate `dimacs` subfolder.

