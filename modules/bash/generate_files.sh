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

# Define parameter values
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

difflrs=("0.001" "0.0001" "0.00001")
alphas=("0.01" "0.1" "1" "10")

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
                                    for alpha in "${alphas[@]}"; do
                                        # Name model
                                        model_name="${dimensions}d_"

                                        if [ "$diffusion" = "True" ]; then
                                            model_name+="diffusion_"
                                        fi

                                        model_name+="difflr_${difflr}_alpha_diff_${alpha}_"
                                                                        
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

                                        mkdir -p $dir_name

                                        # Create configuration files
                                        config_file="$dir_name/config.cfg"
                                        echo "training_data_path=\"$training_data_path\"" >> "$config_file"
                                        echo "dimensions=\"$dimensions\"" >> "$config_file"
                                        echo "species=\"$species\"" >> "$config_file"
                                        echo "density_weight=\"$density_weight\"" >> "$config_file"
                                        echo "uv_layers=$uv_layer" >> "$config_file"
                                        echo "uv_neurons=$uv_neuron" >> "$config_file"
                                        echo "f_layers=$f_layer" >> "$config_file"
                                        echo "f_neurons=$f_neuron" >> "$config_file"
                                        echo "epsilon=$epsilon" >> "$config_file"
                                        echo "points=$point" >> "$config_file"
                                        echo "diffusion=$diffusion" >> "$config_file"
                                        echo "difflr=$difflr" >> "$config_file"
                                        echo "alpha=$alpha" >> "$config_file"

                                        # job1=$(sbatch -p volta-gpu -N 1 -n 1 --mem=40g --qos gpu_access --gres=gpu:1 -t 48:00:00 --output="./slurm/slurm-%j.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/binn/train_binn_comprehensive.py $dir_name")
                                        job1=$(sbatch -p volta-gpu -N 1 -n 1 --mem=40g --qos gpu_access --gres=gpu:1 -t 48:00:00 --output="./$dir_name/slurm-%j.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/binn/train_binn_comprehensive.py $dir_name")

                                        # Extract the job ID from the output (assuming it's in the format "Submitted batch job <job_id>")
                                        job_id1=$(echo "$job1" | awk '{print $4}')

                                        # Submit the second job with dependency on the first job
                                        sbatch --dependency=afterok:$job_id1 -p general -N 1 -n 1 --mem=16g -t 6:00:00 --output="./slurm/slurm-%j.out" --wrap="python /work/users/s/m/smyersn/elston/projects/kinetics_binns/modules/analysis/analysis.py $dir_name"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done