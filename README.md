# Enhancing Predictive Process Monitoring using Semantic Information

This repository contains the scripts and annotated data as described in the paper of Enhancing Predictive Process Monitoring using Semantic Information.

## Data

Log Production is provided in the repository. The remaining can be found in this link:  [logs](https://drive.google.com/file/d/1-6V15CUAsTxVFM2_5H58IBf3HWHEfTIK/view?usp=sharing). 

Each log is annotated in 3 ways. 
- w/o semantics
- w sem (LLM)
- w sem (hardcoded)
  
## Requirements

- Python 3.x+
- Required libraries or dependencies
  
    ```bash
    conda env create -f environment.yml
    ```
    
- Semantic extraction using LLM requires a GPU with at least 24 GB of VRAM

## Run the experiment

1. Download annotated datasets from the link provided before
2. Place the datasets into the 'data/' folder
3. Run following code, replacing the parameters (as described below):
   
```bash
python main.py pattern lm_model data_name
```

###parameters
- pattern: The way to extract business objects and statuses, either "hard" or "llm"'
- lm_model: The huggingface language model to use (e.g. "meta-llama/Meta-Llama-3-8B")
- data_name: The dataset to use (e.g. "production")

e.g. ```bash
python main.py hard meta-llama/Meta-Llama-3-8B production
```

