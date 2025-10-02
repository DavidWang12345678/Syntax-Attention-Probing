# Syntax Attention Probing Using Universal Dependencies

This project explores how much **transformer models capture syntax** by comparing their attention distributions with **Universal Dependencies (UD) treebanks**.

I evaluate both **monolingual** and **multilingual** models:
  - **Monolingual models**: BERT, RoBERTa, DistilBERT
  - **Multilingual model**: mBERT (across English, Spanish, Japanese)

## Key Idea

If attention heads align strongly with **syntactic dependencies** (e.g., focusing on the syntactic head of a word), then the model shows **syntax awareness**.

I measure this with:
  - **Mean mass** → fraction of attention weight assigned to the correct syntactic head.
  - **Precision@k** → whether the head appears in the top-k attended tokens.

## Results

Average mean_mass (attention mass pointing to the syntactic head) for each language × model pair.
![Mean mass by language & model](./results/plots/bar_mean_mass_lang_model.png)

Sentence-level distribution of mean_mass, grouped by language and colored by model.
![Distribution of mean mass](./results/plots/box_mean_mass_lang_model.png)

Each point is a sentence. X = mean_mass, Y = precision@1. Points are colored by language and styled by model.
![Mean mass vs Precision@1 (sentence-level)](./results/plots/scatter_meanmass_prec1.png)
