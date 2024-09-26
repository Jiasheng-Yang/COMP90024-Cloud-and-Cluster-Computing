#!/bin/bash

declare -a configs=("1 1" "1 8" "2 4")

result_file="result.txt"
if [ ! -f "${result_file}" ]; then
    touch "${result_file}"
fi

execute_time_file="execution_times.txt"
if [ ! -f "${execute_time_file}" ]; then
    touch "${execute_time_file}"
fi

slurm_log_folder="slurm_log"
if [ ! -d "${slurm_log_folder}" ]; then
    mkdir "${slurm_log_folder}"
fi


for config in "${configs[@]}"; do
    IFS=' ' read -r -a array <<< "${config}"
    nodes=${array[0]}
    ntasks_per_node=${array[1]}

    temp_script="temp_script_${nodes}_${ntasks_per_node}.slurm"

    # create the 
    cat > ${temp_script} <<EOL
#!/bin/bash
#SBATCH --job-name=run_${nodes}_${ntasks_per_node}
#SBATCH --output=${slurm_log_folder}/%j.out
#SBATCH --error=${slurm_log_folder}/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${ntasks_per_node}

module --ignore_cache load mpi4py

output=\$(srun python main.py)

echo "${nodes} node(s) with ${ntasks_per_node} core(s) result:" >> "${result_file}"
echo "\$output" >> "${result_file}"
echo "" >> "${result_file}"

exec_time=\$(echo "\$output" | grep "Execution Time:" | awk '{print \$3}')
echo "${nodes} node(s) and ${ntasks_per_node} core(s): \$exec_time seconds" >> "${execute_time_file}"
echo "" >> "${result_file}"

# delete the temporary slurm script
rm -- "\$0"
EOL

    # submit the temporary job
    sbatch ${temp_script}
done
