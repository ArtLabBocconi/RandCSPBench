This is the code for our reimplementation of QuerySAT, focusing only on SAT-reduced coloring. The implementation largely follows the same structure as for NeuroSAT.

## Environment installation
To proceed smoothly with the installation, we suggest the use of a virtual environment. The requirements are almost identical to the NeuroSAT environment, so please consult ```algorithms/NeuroSAT/README.md```. After running those commands, please install AdaBelief using ```pip install adabelief-pytorch==0.2.0```


## Training

After the environment is ready, you can train the model using ```train.py```. The code is (mostly) parameterized, so refer to the python script in order to understand what it is possible to tweak. *Please note that at this point you must already have the training and testing data available, otherwise the code will not run successfully. Refer to ```Benchmarks_SAT/datasets``` for details regarding this aspect.* The training commands are essentially identical to NeuroSAT, here are two illustrative examples:

```bash
python train.py assignment ../../datasets/3COL/train-sat --train_splits unknown --valid_dir ../../datasets/3COL/test-sat --valid_splits unknown --epochs 300 --scheduler StepLR --lr_step_size 100 --lr_factor 0.5 --gpu 0 --batch_size 32 --loss unsupervised_2

python train.py assignment ../../datasets/5COL/train-sat --train_splits unknown --valid_dir ../../datasets/5COL/test-sat --valid_splits unknown --epochs 300 --scheduler StepLR --lr_step_size 100 --lr_factor 0.5 --gpu 0 --batch_size 32 --loss unsupervised_2
```

## Evaluation

After the training a model, you will be able to access its saved weights in the ```ckpt/``` folder. The model configuration at the end of the training procedure is also saved. The naming convention of the ckpt files can be found in the training script. Using the saved weights, you can evaluate the model. The evaluation code offers different parameters, for example it is possible to vary the scaling factor in order to alter of the number of iterations. Please consult ```test.py``` for all the possible options. If you wish to download our checkpoints you can do so [here](https://drive.google.com/drive/folders/1GLQZ96rvGYyV0QtEUhyMpP6ZB2AE4K4i?usp=sharing). For the coloring problem, we further have a script that calculates the energy in coloring solution space in order to have a fair comparison with the methods that do in fact work in that space. Therefore, we also have to save the assignment (bitstring) when evaluating. This procedure can significantly burden the evaluation time, especially for large problems (Note that reducing col to sat significantly increases variables and connectivity, and thus computational burden). We therefore provide an additional script named ```test_resumable.py``` that can read already saved results and perform batched saving, in order to allow for easier evaluation on HPC clusters or servers with limited compute time. Here is an example for 3-col:

```bash
python test.py --gpu 0 --K 3 --scaling_factor 2 --num_workers 4 --batch_size 64 --save_every 2 --ckpt_dir ckpt --task col --ckpt_file 3COLSAT_assignment_QuerySAT_unsupervised_2_seed=0_trainS=None_validS=None_perN=True_trainNs=all_13-11-2025-11-36_last.ckpt --dataset_root ../../datasets/ 

python test_resumable.py --gpu 0 --K 3 --scaling_factor 2 --num_workers 4 --batch_size 64 --save_every 2 --ckpt_dir ckpt --task col --ckpt_file 3COLSAT_assignment_QuerySAT_unsupervised_2_seed=0_trainS=None_validS=None_perN=True_trainNs=all_13-11-2025-11-36_last.ckpt --dataset_root ../../datasets/ --resume_from EXISTING_RESULTS_FILE_GOES_HERE

```

After completing the SAT-like evaluation above, you can produce the coloring result files by calling:

```bash
python sat2col_querysat.py --q 3 --cnf_dir ../../datasets/ --csv_dir results_csv/ --csv_result_name COMPLETE_SAT_RESULTS_FROM_ABOVE_TESTING_FILE_GOES_HERE

```


## Credits
This implementation was made from scratch but it largely relies on [G4SATBench](https://github.com/zhaoyu-li/G4SATBench/tree/main). Check out their code and paper if you are interested in GNN-based SAT solvers.
