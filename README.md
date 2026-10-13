# Fine-Tuning Qwen2.5-0.5B-Instruct on Personal X Posts

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/mondayese/qwen-fine-tune)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapter-yellow)](https://huggingface.co/EseMonday/qwen2.5-0.5b-style-adapter)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the dataset preprocessing scripts, LoRA fine-tuning code, and post-mortem lessons for cloning a personal writing style using `Qwen/Qwen2.5-0.5B-Instruct`.



### Links & Assets

* **Run on Kaggle:** [Kaggle Fine-Tuning Notebook](https://www.kaggle.com/code/mondayese/qwen-fine-tune)
* **Hugging Face Model:** [EseMonday/qwen2.5-0.5b-style-adapter](https://huggingface.co/EseMonday/qwen2.5-0.5b-style-adapter)



## Quick Specs

* **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (~495M parameters)
* **Dataset:** 1,930 unique, cleaned posts exported from `tweets.js`
* **Method:** LoRA (Rank $r=8$, Alpha $\alpha=16$, Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`)
* **Trainable Parameters:** 1,081,344 (~0.22% of total model weights)
* **Compute:** Kaggle T4 GPU (~5-8 min training run)
* **Adapter Size:** ~2.2 MB raw FP16 weights (~15.8 MB total repository)



### Pipeline Overview

1. **Data Cleaning (`clean_tweets.py`):** Strips JavaScript variable headers from `tweets.js`, filters retweets/replies/short fragments (<30 chars), removes URLs/mentions, and deduplicates original posts.
2. **Dataset Formatting:** Converts cleaned posts into JSONL records with a static system prompt to force the adapter to learn stylistic patterns (tone, structure, cadence).
3. **Prompt Masking:** Labels system and user prompt tokens with `-100` so cross-entropy loss is computed on the assistant response only.
4. **LoRA Fine-Tuning:** Trains for 3 epochs using `PEFT` and Hugging Face `Trainer` with mixed-precision FP16 and gradient accumulation.




### Quickstart & Usage (`app.py`)

You can run `app.py` directly to load the base Qwen model, attach the adapter from Hugging Face Hub, and generate text in your target writing style.

