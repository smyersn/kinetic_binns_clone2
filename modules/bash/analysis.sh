#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=5g
#SBATCH -t 2-00:00:00
#SBATCH --constraint=rhel8
#SBATCH --output=myjob.out
#SBATCH --mail-type=end
#SBATCH --mail-user=smyersn@ad.unc.edu

source ~/.bashrc
source /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/utils/diff_lists.sh
conda activate binns

mkdir -p runs

# Define your lists of parameter values
repeats=1

training_data_path="/work/users/s/m/smyersn/elston/projects/kinetics_binns/data/2d/random_data.npz"

dimensions=2
species=2

density_weights=("0")

uv_layers=("3") 
uv_neurons=("128")
f_layers=("3")
f_neurons=("32")

epsilons=("0")
points=("0")

# difflrs=("0.000001" "0.0000001" "0.00000001")
difflrs=("0.001")

diffusion="True"

# Create directories for every combination of parameters
for repeat in $(seq 1 $repeats); do
    for density_weight in "${density_weights[@]}"; do
        for uv_layer in "${uv_layers[@]}"; do
            for uv_neuron in "${uv_neurons[@]}"; do
                for f_layer in "${f_layers[@]}"; do
                    for f_neuron in "${f_neurons[@]}"; do
                        for epsilon in "${epsilons[@]}"; do
                            for point in "${points[@]}"; do
                                for difflr in "${difflrs[@]}"; do
                                    # Name model
                                    model_name="${dimensions}d_"

                                    if [ "$diffusion" = "True" ]; then
                                        model_name+="diffusion_leakyrelu_"
                                    fi

                                    model_name+="difflr_${difflr}_"
                                                                    
                                    if diff_lists "${uv_layers[*]}" "3" || diff_lists "${uv_neurons[*]}" "128"; then
                                        model_name+="uvarch_${uv_layer}x${uv_neuron}_"
                                    fi

                                    if diff_lists "${f_layers[*]}" "3" || diff_lists "${f_neurons[*]}" "32"; then
                                        model_name+="farch_${f_layer}x${f_neuron}_"
                                    fi

                                    if diff_lists "${epsilons[*]}" "0" || diff_lists "${points[*]}" "0"; then
                                        model_name+="eps_${epsilon}_pts_${point}_"
                                    fi

                                    if diff_lists "${density_weights[*]}" "0" ; then
                                        model_name+="density_weight_${density_weight}"
                                    fi

                                    if [ "$repeats" -gt 1 ]; then
                                        model_name+="repeat_${repeat}_"
                                    fi
                                
                                    # Create directory
                                    if [ -z "$model_name" ]; then
                                        dir_name="runs/default"
                                    else
                                        dir_name="runs/${model_name:0:-1}"
                                    fi

                                    # Submit the second job with dependency on the first job
                                    sbatch -p general -N 1 -n 1 --mem=16g -t 6:00:00 --output="./slurm/slurm-%j.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/analysis/analysis.py $dir_name"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done