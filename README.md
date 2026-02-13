# Storytelling Generation Pipeline

This repository provides a SLURM-based pipeline for controlled large-scale text generation experiments using Hugging Face language models.
The system is designed for linguistic and stylistic analysis with systematic variation of generation parameters.


# Overview

The pipeline generates multiple text outputs by varying:
- sampling temperature
- stylistic conditioning (“personalities”)
- number of reiterations

All generations are executed on an HPC cluster and are fully reproducible via parameterized scripts.

# Execution

```bash
sbatch generate.sh
```

The script execute something like:

```bash
python generate.py \
  --model_name SemanticAlignment/Llama-3.1-8B-Italian-SAVA-instruct \
  --language it \
  --genre Bibliography \
  --prefix Minerva7B_ItBio \
  --temperatures 0.7 1.0 1.3 \
  --reiterations 25 \
  --personalities "Dacia Maraini" "Gae Aulenti" \
  --output_folder outputs
```

## Parameters

The list of `generate.py` parameters:

| Parameter       | Description                   |
| --------------- | ----------------------------- |
| `model_name`    | Hugging Face model identifier |
| `language`      | Output language               |
| `genre`         | Target genre                  |
| `prefix`        | Output file prefix            |
| `temperatures`  | Sampling temperatures (list)         |
| `reiterations`  | Generations per configuration |
| `personalities` | Stylistic conditioning labels (list) |
| `output_folder` | Output directory              |

## Outputs

The pipeline produces multiple text files organized by generation parameters.
Each output file encodes model configuration, personality, temperature, and iteration index.

## Reproducibility

All generation parameters are explicit and script-controlled

SLURM stdout/stderr logs are preserved

Reproducibility depends on consistent software and hardware configuration

# Intended Use

This repository is intended for:
- Linguistic analysis
- Computational stylistics
- Controlled text generation experiments

# Acknowledgements

This code was created by: Luca Moroni (moroni@diag.uniroma1.it)