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
