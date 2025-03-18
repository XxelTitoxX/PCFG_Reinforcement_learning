# Learning PCFG

## How to run

1. Install the requirements

    ```bash
    conda env create -f environment.yml
    ```

2. Run the train code

    ```bash
    python train.py --directory ../penn_treebank --use_wandb
    ```

3. (Optional) Run the test code

    ```bash
    python test.py --directory ../penn_treebank --result_path log/{your_folder}/result/{your_result}.json
    ```

## Specifying the gpu

You can specify the gpu by using the `--device` flag. e.g., `--device cuda:0`

Example running two training jobs in parallel:

```bash
nohup python train.py --device cuda:0 --directory ../penn_treebank --use_wandb --num_non_terminals 4 --max_productions 800 &
nohup python train.py --device cuda:1 --directory ../penn_treebank --use_wandb --num_non_terminals 6 --max_productions 800 &
```

## Hyperparameters

Feel free to change the following hyperparameters:\
`--num_non_terminals`: number of non-terminals\
`--max_productions:`: max productions

You can try to change the following hyperparameters:\
`--lr`: learning rate\
`--entropy_weight`: entropy weight\
`--hidden_dim`: hidden dimension\
`--num_layers`: number of layers

Reducing the following hyperparameters will probably lead to worse results:\
`--episode_per_batch`: episode per batch\
`num_sentences_per_score`: number of sentences sampled for each scoring

The flag `--use_wandb` indicates whether to use wandb to log the training process.

The flag `--criterion` indicates which criterion to optimize.
i.e., which criterion to use as a scoring function.
You can choose from `f1`, `probabiltiy`, `coverage`.
In our experiments, we found that the criterion `f1` works the best.

You can use the flag `max_len` to only train on a subset of the training data.
It is used to experiment if the model can overfit the training data.

## Parallel sampling, Different RL algorithms

Our code does not support parallel sampling.
Also, we only implement the PPO algorithm.

We figured implementing these features from the ground up would be inefficient.
Rather, using RL libraries would be better.

We have implemented the algorithm using the RL library [tianshou](https://github.com/thu-ml/tianshou). Check the branch
`tianshou`.
