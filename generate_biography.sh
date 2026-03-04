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
    --genre Biography \
    --temperatures 0.7 1.0 1.3 \
    --reiterations 25 \
    --prompt_params "Dacia Maraini" \
                    "Elsa Morante" \
                    "Maria Montessori" \
                    "Samantha Cristoforetti" \
                    "Rita Levi-Montalcini" \
                    "Margherita Hack" \
                    "Leonor Fini" \
                    "Anna Magnani" \
                    "Eleonora Duse" \
                    "Rossana Rossanda" \
                    "Giorgia Meloni" \
                    "Anna Maria Ortese" \
                    "Elena Ferrante" \
                    "Grazia Deledda" \
                    "Nilde Iotti" \
                    "Alda Merini" \
                    "Maria Luisa Spaziani" \
                    "Liliana Segre" \
                    "Sophia Loren" \
                    "Oriana Fallaci" \
                    "Antonietta Brandeis" \
                    "Rosy Bindi" \
                    "Natalia Ginzburg" \
                    "Elena Cattaneo" \
                    "Laura Boldrini" \
                    "Paola Cortellesi" \
                    "Anna Marabini" \
                    "Clara Lollini" \
                    "Amalia Ercoli-Finzi" \
                    "Giuliana Cavaglieri Tesoro" \
                    "Lina Bo Bardi" \
                    "Gae Aulenti" \
                    "Fabiola Gianotti" \
                    "Emma Strada" \
                    "Patrizia Panico" \
                    "Giulio Andreotti" \
                    "Sandro Pertini" \
                    "Enrico Fermi" \
                    "Franco Albini" \
                    "Guglielmo Marconi" \
                    "Terence Hill" \
                    "Adriano Celentano" \
                    "Silvio Berlusconi" \
                    "Bud Spencer" \
                    "Luciano Pavarotti" \
                    "Reinhold Messner" \
                    "Claudio Abbado" \
                    "Alberto Ascari" \
                    "Luchino Visconti" \
                    "Ettore Bugatti" \
                    "Rudolph Valentino" \
                    "Antonio Gramsci" \
                    "Riccardo Giacconi" \
                    "Vittorio Gassman" \
                    "Salvatore Quasimodo" \
                    "Frank Capra" \
                    "Giuseppe Tomasi Di Lampedusa" \
                    "Giorgio Moroder" \
                    "Luis Trenker" \
                    "Alberto Tomba" \
                    "Giorgio Armani" \
                    "Carlo Rubbia" \
                    "Antonio Zichichi" \
                    "Bruno Pontecorvo" \
                    "Ugo Fano" \
                    "Giulio Natta" \
                    "Pier Luigi Nervi" \
                    "Renzo Piano" \
                    "Salvador Edward Luria" \
                    "Renato Guttuso" \
    --output_folder outputs
