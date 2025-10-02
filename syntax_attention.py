import os
import argparse
import random
import subprocess
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# UD treebank handling
KNOWN_UD_REPOS = {
    "en": "https://github.com/UniversalDependencies/UD_English-EWT.git",
    "es": "https://github.com/UniversalDependencies/UD_Spanish-AnCora.git",
    "ja": "https://github.com/UniversalDependencies/UD_Japanese-GSD.git",
}

# Find UD directories and map to language codes.
def find_ud_dirs(ud_base: str) -> Dict[str, str]:
    ud_dirs = {}
    if not os.path.isdir(ud_base):
        return ud_dirs
        
    for entry in os.listdir(ud_base):
        full = os.path.join(ud_base, entry)
        if os.path.isdir(full) and entry.startswith("UD_"):
            files = [f for f in os.listdir(full) if f.endswith(".conllu")]
            if not files:
                continue
                
            joined = " ".join(files).lower()
            if "en_" in joined or "english" in entry.lower() or "_en" in joined or "-en" in joined:
                ud_dirs["en"] = full
            elif "es_" in joined or "spanish" in entry.lower() or "_es" in joined:
                ud_dirs["es"] = full
            elif "ja_" in joined or "japanese" in entry.lower() or "_ja" in joined:
                ud_dirs["ja"] = full
            else:
                parts = entry.split("_", 1)
                if len(parts) > 1:
                    token = parts[1].lower()
                    if token.startswith("english"):
                        ud_dirs.setdefault("en", full)
                    elif token.startswith("spanish"):
                        ud_dirs.setdefault("es", full)
                    elif token.startswith("japanese"):
                        ud_dirs.setdefault("ja", full)
    return ud_dirs

def ensure_ud_treebank(lang: str, ud_base: str) -> Optional[str]:
    ud_dirs = find_ud_dirs(ud_base)
    if lang in ud_dirs:
        return ud_dirs[lang]

    repo = KNOWN_UD_REPOS.get(lang)
    if not repo:
        return None
        
    target_name = os.path.basename(repo).replace(".git", "")
    target_path = os.path.join(ud_base, target_name)
    if os.path.isdir(target_path) and any(f.endswith(".conllu") for f in os.listdir(target_path)):
        return target_path
        
    try:
        print(f"Downloading UD treebank for {lang}...")
        subprocess.run(["git", "clone", "--depth", "1", repo, target_path], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if any(fname.endswith(".conllu") for fname in os.listdir(target_path)):
            return target_path
        else:
            print(f"Download succeeded but no .conllu files found in {target_path}")
            return None
    except Exception as e:
        print(f"Failed to download treebank for {lang}: {e}")
        return None

def read_conllu_sentences(conllu_path: str) -> List[Dict]:
    # Parse CoNLL-U file into sentence dictionaries.
    sentences = []
    try:
        with open(conllu_path, encoding="utf-8") as fh:
            tokens, heads = [], []
            for line in fh:
                line = line.strip()
                if not line:
                    if tokens:
                        sentences.append({"tokens": tokens, "heads": heads})
                        tokens, heads = [], []
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 8:
                    continue
                idx = parts[0]
                if "-" in idx or "." in idx:
                    continue
                token = parts[1]
                head = int(parts[6])
                tokens.append(token)
                heads.append(head)
            if tokens:
                sentences.append({"tokens": tokens, "heads": heads})
    except Exception as e:
        print(f"Error reading {conllu_path}: {e}")
    return sentences

def collect_conllu_files_for_lang(ud_dir: str) -> List[str]:
    files = []
    for fname in os.listdir(ud_dir):
        if fname.endswith(".conllu"):
            files.append(os.path.join(ud_dir, fname))
    return sorted(files)

def sample_ud_sentences(lang_code: str, ud_base: str, n: int, splits=("train","dev","test")) -> List[Dict]:
    ud_dir = ensure_ud_treebank(lang_code, ud_base)
    if not ud_dir:
        raise RuntimeError(f"No UD treebank available for {lang_code}")
        
    conllu_files = collect_conllu_files_for_lang(ud_dir)
    candidate_files = []
    for split in splits:
        for f in conllu_files:
            if split in f.lower():
                candidate_files.append(f)
    if not candidate_files:
        candidate_files = conllu_files

    all_sents = []
    for fpath in candidate_files:
        sents = read_conllu_sentences(fpath)
        if sents:
            random.shuffle(sents)
            all_sents.extend(sents)
        if len(all_sents) >= n:
            break
            
    if len(all_sents) < n:
        print(f"Warning: Only found {len(all_sents)} sentences for {lang_code} (requested {n})")
    return all_sents[:n]

# Tokenization and attention extraction
def tokenize_and_map(tokenizer, tokens: List[str], max_tokens: Optional[int]=None):
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    if not tokens:
        return None, None, None
        
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", 
                   add_special_tokens=True, truncation=True)
    word_ids = enc.word_ids(0)
    subtokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return enc, subtokens, word_ids

def get_attentions_for(tokens: List[str], tokenizer, model, device):
    max_tokens = None
    try:
        max_tokens = max(1, int(tokenizer.model_max_length) - 2)
    except Exception:
        max_tokens = None
        
    enc, subtoks, word_ids = tokenize_and_map(tokenizer, tokens, max_tokens)
    if enc is None:
        return None, None, None
        
    enc = {k: v.to(device) for (k, v) in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    return out.attentions, subtoks, word_ids

def wordid_to_tokmap(word_ids):
    return [w if w is None else int(w) for w in word_ids]

# Attention analysis metrics
def compute_alignment_metrics(attentions, wp_to_tok, tokens: List[str], heads: List[int], exclude_self=False, topk_list=[1,3]):
    if attentions is None:
        return None
        
    if isinstance(attentions, tuple):
        attn_np = np.stack([a[0].cpu().numpy() for a in attentions], axis=0)
    else:
        attn_np = attentions.cpu().numpy()

    num_layers, num_heads, seq_len, _ = attn_np.shape

    # Map tokens to wordpiece indices
    token_to_wps = {}
    for wp_idx, tok_idx in enumerate(wp_to_tok):
        if tok_idx is None: continue
        token_to_wps.setdefault(tok_idx, []).append(wp_idx)

    # Convert 1-indexed heads to 0-based
    token_head = {}
    for i, h in enumerate(heads):
        token_head[i] = None if h == 0 else (h - 1)

    valid_tokens = [i for i in range(len(tokens))
                    if token_head.get(i) is not None and i in token_to_wps and token_head[i] in token_to_wps]

    mean_mass_to_head = np.zeros((num_layers, num_heads))
    mean_mass_to_head_excl = np.zeros((num_layers, num_heads))
    precision_at_k = {k: np.zeros((num_layers, num_heads)) for k in topk_list}

    for l in range(num_layers):
        for h in range(num_heads):
            A = attn_np[l, h]
            if exclude_self:
                A_no_self = A.copy()
                np.fill_diagonal(A_no_self, 0.0)
                row_sums = A_no_self.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                A_norm = A_no_self / row_sums
            else:
                A_norm = A

            masses, masses_excl = [], []
            precisions = {k: [] for k in topk_list}
            for tok in valid_tokens:
                head_tok = token_head[tok]
                S_tok = token_to_wps[tok]
                S_head = token_to_wps[head_tok]
                if not S_tok or not S_head:
                    continue
                    
                masses.append(np.mean([A[s, S_head].sum() for s in S_tok]))
                masses_excl.append(np.mean([A_norm[s, S_head].sum() for s in S_tok]))
                
                for k in topk_list:
                    hits = []
                    for s in S_tok:
                        topk_idx = np.argsort(-A[s])[:k]
                        hits.append(1.0 if any(idx in S_head for idx in topk_idx) else 0.0)
                    precisions[k].append(np.mean(hits))
                    
            mean_mass_to_head[l, h] = np.mean(masses) if masses else 0.0
            mean_mass_to_head_excl[l, h] = np.mean(masses_excl) if masses_excl else 0.0
            for k in topk_list:
                precision_at_k[k][l, h] = np.mean(precisions[k]) if precisions[k] else 0.0

    return {
        "mass": mean_mass_to_head,
        "mass_excl": mean_mass_to_head_excl,
        "precision_at_k": precision_at_k
    }

# Main experiment
def run_experiment(models_list: List[str], multilingual_model: str,
                   langs: List[str], ud_base: str, outdir: str,
                   sample_size: int = 500, use_gpu=False, save_every: int = 200):
    os.makedirs(outdir, exist_ok=True)
    partial_path = os.path.join(outdir, "partial_results.csv")
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    # Tokenization
    lang_to_spacy = {}
    spacy_names = {"en": "en_core_web_sm", "es": "es_core_news_sm", "ja": "ja_core_news_sm"}
    for lang in langs:
        try:
            lang_to_spacy[lang] = spacy.load(spacy_names[lang])
        except Exception as e:
            raise RuntimeError(f"Missing spaCy model for {lang}")

    # Collect sentence samples
    lang_pools = {}
    for lang in langs:
        sents = sample_ud_sentences(lang, ud_base, sample_size)
        if len(sents) < sample_size:
            print(f"Only found {len(sents)} sentences for {lang}")
        lang_pools[lang] = sents

    results_rows = []

    # Process monolingual models
    for model_name in models_list:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=("roberta" in model_name))
            model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
            model.eval()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        pool = lang_pools.get("en", [])
        for i, ex in enumerate(tqdm(pool, desc=f"Processing {model_name} (en)")):
            tokens = ex["tokens"]
            heads = ex["heads"]
            try:
                attns, subtoks, word_ids = get_attentions_for(tokens, tokenizer, model, device)
                if attns is None:
                    continue
                wp_map = wordid_to_tokmap(word_ids)
                met = compute_alignment_metrics(attns, wp_map, tokens, heads, exclude_self=False)
                met_ex = compute_alignment_metrics(attns, wp_map, tokens, heads, exclude_self=True)
                results_rows.append({
                    "model": model_name,
                    "lang": "en",
                    "sentence_idx": i,
                    "mean_mass": float(met["mass"].mean()),
                    "mean_mass_excl_self": float(met_ex["mass_excl"].mean()),
                    "prec@1": float(met["precision_at_k"][1].mean()),
                    "prec@3": float(met["precision_at_k"][3].mean())
                })
            except Exception as e:
                print(f"Error processing {model_name} en idx {i}: {e}")
            if len(results_rows) % save_every == 0:
                pd.DataFrame(results_rows).to_csv(partial_path, index=False)

    # Process multilingual model
    try:
        tok_multi = AutoTokenizer.from_pretrained(multilingual_model)
        model_multi = AutoModel.from_pretrained(multilingual_model, output_attentions=True).to(device)
        model_multi.eval()
    except Exception as e:
        print(f"Failed to load multilingual model: {e}")
        tok_multi = None
        model_multi = None

    if tok_multi is not None and model_multi is not None:
        for lang in langs:
            pool = lang_pools.get(lang, [])
            for i, ex in enumerate(tqdm(pool, desc=f"Processing {multilingual_model} ({lang})")):
                tokens = ex["tokens"]
                heads = ex["heads"]
                try:
                    attns, subtoks, word_ids = get_attentions_for(tokens, tok_multi, model_multi, device)
                    if attns is None:
                        continue
                    wp_map = wordid_to_tokmap(word_ids)
                    met = compute_alignment_metrics(attns, wp_map, tokens, heads, exclude_self=False)
                    met_ex = compute_alignment_metrics(attns, wp_map, tokens, heads, exclude_self=True)
                    results_rows.append({
                        "model": multilingual_model,
                        "lang": lang,
                        "sentence_idx": i,
                        "mean_mass": float(met["mass"].mean()),
                        "mean_mass_excl_self": float(met_ex["mass_excl"].mean()),
                        "prec@1": float(met["precision_at_k"][1].mean()),
                        "prec@3": float(met["precision_at_k"][3].mean())
                    })
                except Exception as e:
                    print(f"Error processing {multilingual_model} {lang} idx {i}: {e}")
                if len(results_rows) % save_every == 0:
                    pd.DataFrame(results_rows).to_csv(partial_path, index=False)

    # Save results
    df = pd.DataFrame(results_rows)
    all_csv = os.path.join(outdir, "all_results.csv")
    df.to_csv(all_csv, index=False)
    print(f"Results saved to {all_csv}")

    # Generate plots
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if not df.empty:
        grp = df.groupby(["model", "lang"]).agg(
            count=("mean_mass", "count"),
            mean_mass_mean=("mean_mass", "mean"),
            mean_mass_std=("mean_mass", "std"),
            prec1_mean=("prec@1", "mean"),
            prec1_std=("prec@1", "std"),
        ).reset_index()

        grp["mean_mass_sem"] = grp["mean_mass_std"] / np.sqrt(grp["count"].replace(0, np.nan))
        grp["prec1_sem"] = grp["prec1_std"] / np.sqrt(grp["count"].replace(0, np.nan))
        grp[["mean_mass_sem", "prec1_sem"]] = grp[["mean_mass_sem", "prec1_sem"]].fillna(0.0)

        # Generate plots
        pivot_mean = grp.pivot(index="lang", columns="model", values="mean_mass_mean").fillna(0.0)
        pivot_sem = grp.pivot(index="lang", columns="model", values="mean_mass_sem").fillna(0.0)

        langs = list(pivot_mean.index)
        models = list(pivot_mean.columns)
        n_langs = len(langs)
        n_models = len(models)
        x = np.arange(n_langs)
        bar_width = 0.8 / max(1, n_models)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, model in enumerate(models):
            ys = pivot_mean[model].values
            yerr = pivot_sem[model].values
            ax.bar(x + i * bar_width, ys, bar_width, label=model, yerr=yerr, capsize=4)
        ax.set_xticks(x + (n_models - 1) * bar_width / 2)
        ax.set_xticklabels(langs)
        ax.set_ylabel("mean_mass")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "bar_mean_mass_lang_model.png"), dpi=200)
        plt.close(fig)

        # Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="lang", y="mean_mass", hue="model")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "box_mean_mass_lang_model.png"), dpi=200)
        plt.close()

        # Scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="mean_mass", y="prec@1", hue="lang", style="model", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "scatter_meanmass_prec1.png"), dpi=200)
        plt.close()

        print(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ud_base", type=str, default="./data", help="Path containing UD treebanks")
    parser.add_argument("--outdir", type=str, default="./results")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--save-every", type=int, default=200, help="Save partial results every N sentences")
    args = parser.parse_args()

    models_list = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
    multilingual_model = "bert-base-multilingual-cased"
    langs = ["en", "es", "ja"]

    run_experiment(models_list, multilingual_model, langs, args.ud_base, args.outdir,
                   sample_size=args.sample_size, use_gpu=args.use_gpu, save_every=args.save_every)
