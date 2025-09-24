# TODO

## Environment installation

To proceed smoothly with the installation, we suggest the use of a virtual environment. The requirements are almost identical to the NeuroSAT environment, so please consult ```algorithms/NeuroSAT/README.md```.

## Training

After the environment is ready, you can train the model using ```train.py```. The code is (mostly) parameterized, so refer to the python script in order to understand what it is possible to tweak. *Please note that at this point you must already have the training and testing data available, otherwise the code will not run successfully. Refer to ```Benchmarks_SAT/datasets``` for details regarding this aspect.*

As illustrative examples, here is how you can train NeuroSAT on our (complete) 3SAT bechmark, using the proposed scaling regime with supervised and unsupervised loss functions:

```bash
python train.py assignment ../../datasets/3SAT/sc/train-final --train_splits unknown --valid_dir ../../datasets/3SAT/sc/test-final --valid_splits unknown --train_label_file ../../datasets/3SAT/sc/train_labels-final.csv --valid_label_file ../../datasets/3SAT/sc/test_labels-final.csv --epochs 300 --scheduler StepLR --lr_step_size 100 --lr_factor 0.5 --gpu 0 --batch_size 32 --loss unsupervised_2

python train.py assignment ../../datasets/3SAT/sc/train-final --train_splits unknown --valid_dir ../../datasets/3SAT/sc/test-final --valid_splits unknown --train_label_file ../../datasets/3SAT/sc/train_labels-final.csv --valid_label_file ../../datasets/3SAT/sc/test_labels-final.csv --epochs 300 --scheduler StepLR --lr_step_size 100 --lr_factor 0.5 --gpu 0 --batch_size 32 --loss supervised

```

## Evaluation

After the training a model, you will be able to access its saved weights in the ```ckpt/``` folder. The weights are saved based on performance (accuracy) on the testing set, returning the top two configurations in terms of generalization. Finally, the model configuration at the end of the training procedure is also saved. The naming convention of the ckpt files can be found in the training script, in lines 59-62. Using the saved weights, you can evaluate the model. The evaluation code offers different parameters, for example it is possible to vary the scaling factor in order to alter of the number of iterations. Please consult ```test.py``` for all the possible options. Below we provide the commands for evaluating NeuroSAT on the 3SAT (supervised and unsupervised) and 4SAT benchmarks for 2N iterations, as presented in the paper. You can download the checkpoints [here](https://drive.google.com/drive/folders/1YwDuuDh023-XuMco54c1d9PvsO5jQr9v?usp=sharing).The evaluation procedure leads to the generation of a csv file containing information on the solving of the testing examples, which you can then explore, for example, using the (Jupyter) notebooks in the ```notebooks/``` directory.

```bash
python test.py --gpu 0 --K 3 --scaling_factor 2 --ckpt_dir ckpt --ckpt_file 3SAT_assignment_neurosat_supervised_seed=0_trainS=None_validS=None_perN=True_19-07-2024-22-27_epoch=199.ckpt

python test.py --gpu 0 --K 3 --scaling_factor 2 --ckpt_dir ckpt --ckpt_file 3SAT_assignment_neurosat_unsupervised_2_seed=0_trainS=None_validS=None_perN=True_trainNs=all_09-09-2024-23-51_epoch=125.ckpt

python test.py --gpu 0 --K 4 --scaling_factor 2 --ckpt_dir ckpt --ckpt_file 4SAT_assignment_neurosat_unsupervised_2_seed=0_trainS=None_validS=None_perN=True_trainNs=all_05-12-2024-19-43_continuation_from_26-10-2024-19-58_epoch=281.ckpt
```

## Credits
This implementation is largely based on [G4SATBench](https://github.com/zhaoyu-li/G4SATBench/tree/main). Check out their code and paper if you are interested in GNN-based SAT solvers.
