# Verifiable Boosted Tree Ensembles

This repository contains the implementation of the training algorithm for large-spread boosted tree ensembles, based on LightGBM, and the efficient robustness verification algorithm CARVE-GBM presented by Calzavara et. al. in their research paper Verifiable Boosted Tree Ensembles accepted at the [46th IEEE Symposium on Security and Privacy 2025 (IEEE S&P 2025)](https://sp2025.ieee-security.org/).

## Artifact organization

The artifact is organized in the following folders:

- the **datasets/** folder contains the datasets used to train the large-spread boosted ensembles. See the "Obtain the datasets" section for more details about its subfolders.
- the **src/** folder contains:
	- the **training/** folder that contains all the scripts useful for training the models. It contains also a reference to the [modified version of LightGBM](https://github.com/LorenzoCazzaro/LightGBM-verifiable-boosted-tree-ensembles) that supports the enforcing of the large-spread condition.
	- the **verification/** folder that contains all the scripts useful to verify the robustness of the models. It contains the code of the following verifiers:
		- CARVE-GBM in the **carvegbm/** folder.
		- A reference to our modified version of [SILVA](https://github.com/LorenzoCazzaro/silva) in the **silva/** folder.

## Download the repo

Download the repo using `git clone <repo_link> --recursive` to download also the submodules.

## System configuration

In the paper you may find some details of the system in which we run the experiments. Here we report some details about the software. Our system uses:
<ul>
	<li> python (3.8) </li>
	<li> some python modules: scikit-learn (1.0.2), numpy (1.22.3), argparse (1.1), pandas (1.4.2), matplotlib (3.5.1), lightgbm (4.1.0)
	<li> g++ (9.4.0) </li>
	<li> make (4.2.1) </li>
</ul>

You can use **docker** to run a container running Ubuntu and with all the dependensies installed. Use the script *start_docker_container.sh* in the main folder to build and run the docker. It requires to have installed **docker**.

## Obtain the datasets

You can produce the training sets and test sets used in our experimental evaluation by executing the bash script <em>download_and_split_datasets.sh</em> in the <em>src/datasets_utils</em> folder.

If you want to use another dataset, you have to create the folder *datasets/<dataset_name>/* and the following folders in it:

- *dataset/*, that will contain the training_set, test_set e validation_set.</li>
- *models/*, *models/gbdt/* and *models/gbdt_lse/*, that will contain the trained GDBT models and large-spread boosted ensembles.</li>

The datasets in the *datasets/<dataset_name>/dataset/* must be named as follows:

- *training_set_normalized* for the training set;</li>
- *test_set_normalized* for the test set;</li>
- *validation_set_normalized* for the validation set obtained by dividing the entire training set in the (sub)-training set and validation set.</li>

## Compile the tools

### Training - LightGBM

See the README.md in the <em>src/lightgbm</em> folder.

### Verification - CARVE-GBM

See the README.md in the <em>src/carvegbm</em> folder.

### Verification - SILVA

See the README.md in the <em>src/silva</em> folder.

## Use the tools

### Training - LightGBM

See the README.md in the <em>src/lightgbm</em> folder.

Example: `./lightgbm/lightgbm config=train.conf boosting=gbdt data=../../datasets/mnist26/dataset/training_set_normalized.csv valid=../../datasets/mnist26/dataset/validation_set_normalized.csv num_trees=500 num_leaves=16 k=0.01 seed=0 output_model=../../datasets/mnist26/models/gbdt_lse/lightgbm_lse_best_0_16_inf_0.01_subflsc_-1.txt p=inf learning_rate=0.1 early_stopping_round=50 feature_fraction=1 verbose=-1` (run it in the <em>src/training/lightgbm/build</em> folder).

### Verification - CARVE-GBM

See the README.md in the src/carve folder.

Example: `./verify -i ../../../datasets/mnist26/models/gbdt_lse/<model_name> -t ../../../datasets/mnist26/dataset/test_set_normalized.csv -p inf -k 0.01 -ioi -1` (run it in the <em>src/verification/carvegbm/build</em> folder).

### Verification - SILVA

See the README.md in the <em>src/silva</em> folder.

Example: `./silva/src/silva ../datasets/mnist26/models/gbdt/<model_name> ../datasets/mnist26/dataset/test_set_normalized.csv --perturbation l_inf 0.01 --index-of-instance -1 --voting softargmax` (run it in the <em>src/verification/silva</em> folder).

## Basic test

After compiling all the tools, you can run this simple test to check that everything works fine:

TBD

## Generate experimental results

TBD

## Credits

If you use this artifact in your work, please add a reference/citation to our paper. You can use the following BibTeX entry:

TBD

## Support

If you want to ask questions about the artifact and notify bugs, feel free to contact us by sending an email to lorenzo.cazzaro@unive.it.