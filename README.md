# LayerSkipping

Setup environment:

```
# remember to change transformers==4.45.2 in requirements.txt
$ conda create --name layer_skip python=3.10
$ conda activate layer_skip
$ pip install -r requirements.txt
```

In the command-line run huggingface-cli login, and you will be prompted to provide the token you have obtained: \*\*\*

If the token doesn't work then you have to make your own --><br/>
Access models: In order to observe speedup, you need to access LLMs that have been trained using the LayerSkip recipe: https://huggingface.co/facebook/layerskip-llama2-7B

Visit the model's corresponding link above, make sure you are logged on the HuggingFace website with your account.
Fill the request form and submit it. Approval may take a while and you should receive an email notification to notify you that permission to the model is granted.
Follow the steps here to obtain a user access token.
In the command-line run huggingface-cli login, and you will be prompted to provide the token you have obtained in Step 3.

To train:

```
python finetune_gates.py
```

# LayerSkipping Early Exit
The set up environment is the same as in LayerSkipping 

To run the early exit file:
Select the specific model for training in checkpoint = "{model_name}" in early_exit.py
Then run

```
python early_exit.py
```

Afterwards you should get the csv files containing information about the evaluation of each model. This information can be visualized by running the early exit cells

```
visualization.ipynb
```

# LayerSkipping Evaluations
The set up environment is the same as in LayerSkipping 

To run the evaluation script:


```
chmod +x run_all.sh
./run_all.sh
```

Afterwards you should get the log files containing information about the evaluation of each model. This information can be visualized by running the evaluation cells in 

```
visualization.ipynb
```

