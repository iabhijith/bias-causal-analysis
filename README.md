# CMA Framework for reducing bias in LLMs


## Identification

### DiffMask+
- Run DiffMask+ using ```python discover.py```
- Configuration for DiffMask+ can be changed as required in ```configuration/diffmask.yaml```
- Alternatively, can be provided as command line arguments (hydra)
    - For example, ```python discover.py attn_heads=20``` to run DiffMask+ for selecting top 20 attention heads
- The output of DiffMask+ is stored in the directory as specified in ```results_path``` in ```configuration/diffmask.yaml```

### CMA
- Run CMA using ```python discover_cma.py```
- Configuration for CMA can be changed as required in ```configuration/cma.yaml```
- Alternatively, can be provided as command line arguments (hydra)
    - For example, ```python discover_cma.py attn_heads=20```

    
## Mitigation
- For fine-tuning the model only on identified components using BUG balanced dataset, run ```python mitigate.py```
- Configuration for mitigation can be changed as required in ```configuration/tuner.yaml```
- Alternatively, can be provided as command line arguments (hydra)
    - For example, ```python mitigate.py components=acdc```
- The fine-tuned model is saved to the directory as specified in ```results_path``` in ```configuration/tuner.yaml```

## Evaluation 

- For evaluating the fine-tuned model on the BUG balanced dataset, run ```python evaluate.py```
- TODO: Clean up the code and add more documentation
