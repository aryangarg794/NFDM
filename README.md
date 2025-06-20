## NFDM Reproduction 

--------------
To run the repo install `uv` and then run 
```bash
uv sync
```

If using CUDA then you need to install the `torch+cu118` package manually using `uv pip install`.


-------- 
To run the repo just run:

```bash
python3 src/nfdm/main.py
```
## 🧾 Command-Line Arguments

| Argument             | Short | Type     | Default      | Description                                      |
|----------------------|-------|----------|--------------|--------------------------------------------------|
| `--epochs`           | `-e`  | `int`    | `100`        | Number of training epochs                        |
| `--batch_size`       | `-b`  | `int`    | `128`        | Batch size for training                          |
| `--lr`               | `-lr` | `float`  | `4e-4`       | Learning rate                                    |
| `--save`             | `-s`  | `flag`   | `False`      | Save model after training                        |
| `--device`           | `-d`  | `str`    | `'cpu'`      | Device to use: `'cpu'` or `'cuda'`               |
| `--test`             | `-t`  | `flag`   | `False`      | Run in test (inference) mode                     |
| `--amp`              | `-a`  | `flag`   | `False`      | Enable Automatic Mixed Precision (AMP)           |
| `--seed`             |       | `int`    | `None`       | Set seed for reproducibility                     |
| `--dir`              |       | `str`    | `None`       | Path to load a saved model                       |
| `--model`            | `-m`  | `str`    | `'nfdm'`     | Model type: `nfdm`, `ddpm`, or `ae`              |

Make changes accordingly to run your experiments. 

------- 
#### Project Structure

`examples`: contains generated samples from the frameworks
`models`: contains saved checkpoints or full models for testing and generation
`src/nfdm`: contains the base code
`src/nfdm/models`: code for the models 
`src/nfdm/utils`: dataloading code etc 

#### References
--------
[1] Bartosh, G., Vetrov, D., & Naesseth, C. A. (2024). Neural Flow Diffusion Models: Learnable Forward Process for Improved Diffusion Modelling. arXiv [Stat.ML]. Retrieved from http://arxiv.org/abs/2404.12940
