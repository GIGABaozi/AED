# Alignment-Enhanced Decoding (AED)

This repository is the implementation of the paper: Alignment-Enhanced Decoding: Defending via Token-Level Adaptive Refining of Probability Distributions. 

In this paper, we present a novel defense that employs adaptive decoding to address the root causes of jailbreak issues.😊

## Abstract
Large language models are susceptible to jailbreak attacks, which can result in the generation of harmful content. While prior defenses mitigate these risks by perturbing or inspecting inputs, they ignore competing objectives, the underlying cause of alignment failures. In this paper, we propose Alignment-Enhanced Decoding (AED), a novel defense that employs adaptive decoding to address the root causes of jailbreak issues. We first define the Competitive Index to quantify alignment failures and utilize feedback from self-evaluation to compute post-alignment logits. Then, AED adaptively combines Competitive Index and post-alignment logits with the original logits to obtain harmless and helpful distributions. Consequently, our method enhances safety alignment while maintaining helpfulness. We conduct experiments across five models and four common jailbreaks, with the results validating the effectiveness of our approach.

## Pipeline
AED has 3 steps: Step 1 involves obtaining the probability distribution of the next token; Step 2 computes the Competitive Index, which reflects the degree of competitions; and Step 3 realigns the distribution to ensure a safe and ethical response. More detail could be found in our paper.😄 ![Alt text](./figs/pipeline.png) 

### Tested Models
|           LLMs            | AED |
|:-------------------------:|:--------------------------:|
|    Llama-2-7b-chat-hf     |             ✅              |
|      vicuna-7b-v1.5       |             ✅              |
| Meta-Llama-3-8B-Instruct  |             ✅              |
| Gemma-1.1-7B-it  |             ✅              |
| Guanaco-7B  |             ✅              |

### Defended Jailbreaks

|           LLMs            | AED |
|:-------------------------:|:--------------------------:|
|    GCG     |             ✅              |
|       AutoDAN      |             ✅              |
| ICA  |             ✅              |
| Refusal_Sup.  |             ✅              |

### Harmless Benchmark

|           LLMs            | AED |
|:-------------------------:|:--------------------------:|
|    MMLU     |             ✅              |
|       GMS8K      |             ✅              |
| Alpaca  |             ✅              |



## Implementation

The main codes can be found at  ```main.ipynb```. 

```       
defence(model_name, model, tokenizer, device, dataset, dataset_bias, begin, end, filename)
```

If you wan to try other models, you can just change the  ```model_name ``` to ```vicuna```,```llama3```, ```gemma``` or ```guanaco```. Don't forget to change the path of your model.
```
model_name = "llama"
model_path = "../llama2-7b-chat"
```

if you try to try other dataset, you can change ```dataset``` to the one you want and add the coresponding pre-process in function ```get_data``` in ```utilz.py```

```
dataset = ["gcg"]

...

def get_data(dataset_name, ADD_SYSTEM_PROMPT=True, prompt='dafult'):
```

---
