import subprocess

def run(cmd):
    print(f"Running command: {cmd}", flush=True)

    # Execute the command and continuously print its output
    subprocess.call(cmd, shell=True)

# Define the command line string to execute
run("python3 generate_plots_hyperparameters.py RAM_accuracies.csv")
run("python3 generate_plots_BO.py RAM_accuracies_BO.csv")
run("python3 generate_plots_noise.py RAM_accuracies_noise.csv")
run("python3 generate_plots_mem_based_other_datasets.py RAM_accuracies_mem_based_other_datasets.csv")