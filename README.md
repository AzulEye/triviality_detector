# Triviality Analysis of Text Using Language Models

## Overview

This script calculates and compares the **per-token perplexity** of input text across multiple language models, and generates visual plots to assess the **triviality** of text. Triviality is computed as the ratio of log perplexities from two models—typically a smaller and a larger model—on the same text.

The workflow involves:
1. Loading pre-trained language models.
2. Processing text files to compute per-token perplexity.
3. Calculating summary statistics (mean, standard error).
4. Generating comparison plots between the models' perplexities.
5. Saving results in pickle files.

## Requirements

### Dependencies
Install the following dependencies:
```bash
pip install torch==2.2.1 transformers==4.44.2 matplotlib seaborn scipy numpy argparse
```

## Usage

### Input Arguments
The script requires several input arguments that can be passed through the command line.

- `--k`: Start calculating perplexity from the `k-th` token (default: `100`).
- `--max_words`: The maximum number of words to process per file (default: `1000`).
- `--cache_dir`: Directory where the models are cached (default: `~/`).
- `--plot_dir`: Directory where generated plots will be saved (default: `plots`).
- `--pickle_dir`: Directory where pickle files containing results will be saved (default: `pickles`).

### Text Input
- Text files are expected in a directory named `text_chunks`.
- The script processes each file in this directory, calculates the perplexity for each model, and generates corresponding plots.

### Example Command
```bash
python triviality_analysis.py --k 100 --max_words 1000 --cache_dir "./model_cache" --plot_dir "./plots" --pickle_dir "./pickles"
```

### Output
- Plots are saved in the `plots` directory, showing the comparison of perplexities between two models and the computed **Triviality Score**.
- Results are also saved as pickle files in the `pickles` directory.

## License
This project is licensed under the MIT License.

## Author
Aharon Azulay

aazuleye@gmail.com

For further details, feel free to explore the code and modify it based on your specific needs.