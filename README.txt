# Clone the repo
git clone https://github.com/martinmathew/assignment_4_markov_decision_process.git

# Change Directory
cd assignment_4_markov_decision_process

# create env
conda env update --prefix ./env --file environment.yml  --prune

# activate env
conda activate path to env\env

# For Charts related to Mountain Car
python mountain_car_orig.py
python mountain_car_experiment.py


# For Charts related to Frozen lakes
python frozen_lake_experiments.py

