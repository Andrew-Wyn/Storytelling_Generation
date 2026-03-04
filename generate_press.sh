#!/bin/bash
#SBATCH --job-name=storytelling                         # Job name
#SBATCH --output=%x-%j.out                              # Name of stdout output file
#SBATCH --error=%x-%j.err                               # Name of stderr error file

#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks-per-node=1                             # number of tasks per node
#SBATCH --cpus-per-task=32                              # number of threads per task
#SBATCH --time 24:00:00                                 # format: HH:MM:SS
#SBATCH --gres=gpu:1                                    # number of gpus per node

#SBATCH -A ACCOUNT_NAME                                 # account to charge
#SBATCH -p boost_usr_prod                               # partition to execute

# Load necessary modules
# module load python
# module load cuda/12.1

# Set up HuggingFace
# export HF_HOME="PATH_TO_CACHE"
# export HF_DATASETS_CACHE="PATH_TO_CACHE"
# export HF_HUB_OFFLINE=0
# export HF_DATASETS_OFFLINE=0
# export TRANSFORMERS_OFFLINE=0

cd  path/to/this/folder

# # load virtual environment
source .env/bin/activate


python generate.py --model_name SemanticAlignment/Llama-3.1-8B-Italian-SAVA-instruct \
    --language it \
    --genre Press_Release \
    --temperatures 0.7 1.0 1.3 \
    --reiterations 25 \
    --prompt_params "GPT-4" \
                    "GPT-3.5" \
                    "Dall-E" \
                    "Gemini" \
                    "BERT" \
                    "RoBERTa" \
                    "Bard" \
                    "Mistral 7B" \
                    "Claude" \
                    "Claude 3" \
                    "Dante AI" \
                    "GPT-Neo" \
                    "Galactica" \
                    "BloombergGPT" \
                    "Gemma" \
                    "Chinchilla" \
                    "Megatron" \
                    "CamemBERT" \
                    "GPT-4.5" \
                    "Libra" \
                    "GeniusGPT" \
                    "Sigma 1.5" \
                    "Semantech" \
                    "Meta" \
                    "Cogni AI" \
    --output_folder outputs
