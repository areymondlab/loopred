#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
from Bio import SeqIO
import pyfaidx
import pybedtools
import regex as re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
import logging
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import umap
from statsmodels.stats.multitest import multipletests
import itertools
import random
from numba import njit
import gzip
import scikit_posthocs as sp
import shap
import argparse
import sys
import json
try:
    from gensim.models import Word2Vec, KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Gensim not available. Word2Vec embeddings will not be supported.")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("dna_structure_analysis.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class StructureData:
    sequences: List[str]
    labels: List[str]
    structure_type: str
    intervals: Optional[List[Tuple[str, int, int]]] = None

@njit(cache=True)
def _process_sequence_jit(seq_encoded: np.ndarray) -> np.ndarray:
    length = len(seq_encoded)
    k_freq_sizes = [4, 16, 64, 256]
    results = np.zeros(4 + 4 + sum(k_freq_sizes), dtype=np.float64)
    if length == 0: 
        return results

    base_counts = np.zeros(4, dtype=np.int64)
    kmer_counts = [np.zeros(s, dtype=np.int64) for s in k_freq_sizes]
    run_counters = np.zeros(4, dtype=np.int64)
    max_runs = np.zeros(4, dtype=np.int64)

    for i in range(length):
        base = seq_encoded[i]
        base_counts[base] += 1
        
        for b_idx in range(4):
            if base == b_idx: 
                run_counters[b_idx] += 1
            else:
                if run_counters[b_idx] > max_runs[b_idx]: 
                    max_runs[b_idx] = run_counters[b_idx]
                run_counters[b_idx] = 0
                
        for k_idx in range(4):
            k = k_idx + 1
            if i >= k - 1:
                kmer_idx = 0
                for j in range(k): 
                    kmer_idx = kmer_idx * 4 + seq_encoded[i - k + 1 + j]
                kmer_counts[k_idx][kmer_idx] += 1
                
    for b_idx in range(4):
        if run_counters[b_idx] > max_runs[b_idx]: 
            max_runs[b_idx] = run_counters[b_idx]

    results[0:4] = base_counts
    results[4:8] = max_runs
    offset = 8
    for k_idx in range(4):
        k = k_idx + 1
        total_kmers = length - k + 1
        if total_kmers > 0: 
            results[offset:offset+k_freq_sizes[k_idx]] = kmer_counts[k_idx] / total_kmers
        offset += k_freq_sizes[k_idx]
    return results

class BedFileProcessor:
    def __init__(self, reference_genome_path: str):
        logger.info(f"Initializing BedFileProcessor with reference genome: {reference_genome_path}")
        self.genome = pyfaidx.Fasta(reference_genome_path)
        
    def load_bed_file(self, bed_path: str, stype: str) -> pybedtools.BedTool:
        logger.info(f"Loading BED file for '{stype}' from {bed_path}")
        return pybedtools.BedTool(bed_path)
        
    def merge_and_deduplicate_beds(self, bed_files: Dict[str, str]) -> Dict[str, pybedtools.BedTool]:
        logger.info("Merging and sorting BED files for all types.")
        return {stype: self.load_bed_file(path, stype).sort().merge() 
                for stype, path in bed_files.items()}
                
    def extract_sequences_from_beds(self, processed_beds: Dict[str, pybedtools.BedTool]) -> List[StructureData]:
        data_list = []
        for stype, bed in processed_beds.items():
            sequences, intervals = [], []
            logger.info(f"Extracting sequences for structure type: {stype}")
            for i in tqdm(bed, desc=f"Extracting {stype}", leave=False):
                if i.start < i.end and i.chrom in self.genome:
                    seq = str(self.genome[i.chrom][i.start:i.end])
                    if seq and (seq.count("N") / len(seq) < 0.1):
                        sequences.append(seq.upper())
                        intervals.append((i.chrom, i.start, i.end))
            if sequences:
                logger.info(f"Extracted {len(sequences)} valid sequences for {stype}.")
                data_list.append(StructureData(sequences, [stype]*len(sequences), stype, intervals))
        return data_list

class AdvancedDNAFeatureExtractor:
    def __init__(self):
        self.motif_patterns = self._compile_motif_patterns()
        self.k_freq_sizes = [4, 16, 64, 256]
        self.jit_feature_names = (
            ["count_A", "count_T", "count_G", "count_C"] +
            ["max_run_A", "max_run_T", "max_run_G", "max_run_C"] +
            [f"k{k}_{''.join(p)}" for k in range(1,5) 
             for p in itertools.product("ATGC", repeat=k)]
        )
        
    def _compile_motif_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "cpg_islands": re.compile(r"CG"), 
            "tata_box": re.compile(r"TATAAA"), 
            "caat_box": re.compile(r"CCAAT"),
            "gc_sp1_box": re.compile(r"GGGCGG"), 
            "initiator": re.compile(r"[CT]{2}A[ATGC][AT][CT]{2}"),
            "dpe": re.compile(r"[AG]G[AT]CGTG"), 
            "bre": re.compile(r"[GC]{2}[AG]CGCC"),
            "polya_signal": re.compile(r"AATAAA"), 
            "kozak": re.compile(r"GCC[AG]CCATGG"),
            "palindrome_6": re.compile(r"(\w)(\w)(\w)\3\2\1"), 
            "palindrome_8": re.compile(r"(\w)(\w)(\w)(\w)\4\3\2\1"),
            "at_rich_6": re.compile(r"[AT]{6,}"), 
            "gc_rich_6": re.compile(r"[GC]{6,}"),
            "purine_rich": re.compile(r"[AG]{8,}"), 
            "pyrimidine_rich": re.compile(r"[CT]{8,}"),
            "e_box": re.compile(r"CA[ATGC]{2}TG"), 
            "ap1_site": re.compile(r"TGAGTCA"),
            "z_dna_ca": re.compile(r"(CA){4,}"), 
            "g_quadruplex": re.compile(r"G{3,}[ATCGN]{1,7}G{3,}[ATCGN]{1,7}G{3,}[ATCGN]{1,7}G{3,}"),
            "ca_repeat": re.compile(r"(CA){3,}"),
        }
        
    def _encode_sequence(self, sequence: str) -> np.ndarray:
        mapping = {"A": 0, "T": 1, "G": 2, "C": 3}
        return np.array([mapping.get(b, 0) for b in sequence], dtype=np.int8)
        
    def extract_motif_features(self, sequence: str, length: int) -> Dict[str, float]:
        if length == 0: 
            return {f"motif_{name}_density": 0.0 for name in self.motif_patterns}
        return {f"motif_{name}_density": len(pattern.findall(sequence)) / length 
                for name, pattern in self.motif_patterns.items()}
                
    def extract_derived_features(self, jit_results: np.ndarray, length: int) -> Dict[str, float]:
        if length == 0: 
            return {}
            
        features = {}
        a, t, g, c = jit_results[0], jit_results[1], jit_results[2], jit_results[3]
        features["gc_content"] = (g + c) / length if length > 0 else 0
        features["gc_skew"] = (g - c) / max(1, g + c)
        
        offset = 8
        for k_idx, k_size in enumerate(self.k_freq_sizes):
            k = k_idx + 1
            freqs = jit_results[offset:offset+k_size]
            entropy = -np.sum(freqs * np.log2(freqs + 1e-100))
            features[f"kmer_complexity_{k}"] = entropy / (k * 2.0) if k > 0 else 0
            offset += k_size
        return features
        
    def extract_all_features(self, sequence: str) -> Dict[str, float]:
        seq_upper = sequence.upper()
        length = len(seq_upper)
        jit_results = _process_sequence_jit(self._encode_sequence(seq_upper))
        all_features = dict(zip(self.jit_feature_names, jit_results))
        all_features["length"] = float(length)
        all_features.update(self.extract_derived_features(jit_results, length))
        all_features.update(self.extract_motif_features(seq_upper, length))
        return all_features

class DNAEmbeddingExtractor:
    def __init__(self, embedding_dim: int = 100, kmer_size: int = 6, 
                 w2v_model_path: Optional[str] = None, embedding_method: str = "bow"):
        self.embedding_dim = embedding_dim
        self.kmer_size = kmer_size
        self.embedding_method = embedding_method
        self.w2v_model = None
        
        if embedding_method == "w2v":
            if not GENSIM_AVAILABLE:
                raise ImportError("Gensim is required for Word2Vec embeddings. Install with: pip install gensim")
            if w2v_model_path is None:
                raise ValueError("w2v_model_path must be provided when using Word2Vec embeddings")
            self._load_w2v_model(w2v_model_path)
        
        logger.info(f"Initialized DNAEmbeddingExtractor with method='{embedding_method}', "
                   f"k-mer size {kmer_size} and embedding dim {embedding_dim}")
    
    def _load_w2v_model(self, model_path: str):
        """Load Word2Vec model from file."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Word2Vec model file not found: {model_path}")
        
        try:
            # Try loading as KeyedVectors first (more common format)
            if model_path.suffix in ['.vec', '.txt']:
                self.w2v_model = KeyedVectors.load_word2vec_format(str(model_path), binary=False)
            elif model_path.suffix == '.bin':
                self.w2v_model = KeyedVectors.load_word2vec_format(str(model_path), binary=True)
            else:
                # Try loading as full Word2Vec model
                self.w2v_model = Word2Vec.load(str(model_path))
                self.w2v_model = self.w2v_model.wv  # Extract KeyedVectors
            
            logger.info(f"Loaded Word2Vec model with {len(self.w2v_model.index_to_key)} k-mers, "
                       f"vector size {self.w2v_model.vector_size}")
            
            # Update embedding_dim to match model
            if self.embedding_dim != self.w2v_model.vector_size:
                logger.warning(f"Updating embedding_dim from {self.embedding_dim} to "
                             f"{self.w2v_model.vector_size} to match Word2Vec model")
                self.embedding_dim = self.w2v_model.vector_size
                
        except Exception as e:
            raise ValueError(f"Failed to load Word2Vec model from {model_path}: {e}")
    
    def _extract_kmers_from_sequence(self, sequence: str) -> List[str]:
        """Extract k-mers from a sequence."""
        if len(sequence) < self.kmer_size:
            return []
        
        seq_upper = sequence.upper()
        kmers = []
        for i in range(len(seq_upper) - self.kmer_size + 1):
            kmer = seq_upper[i:i+self.kmer_size]
            if "N" not in kmer:
                kmers.append(kmer)
        return kmers
    
    def extract_w2v_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Extract Word2Vec embeddings for sequences."""
        logger.info("Generating Word2Vec k-mer embeddings...")
        embeddings = np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)
        
        for i, seq in enumerate(tqdm(sequences, desc="W2V Embeddings", leave=False)):
            kmers = self._extract_kmers_from_sequence(seq)
            if not kmers:
                continue
            
            # Get embeddings for k-mers that exist in the model
            valid_embeddings = []
            for kmer in kmers:
                if kmer in self.w2v_model:
                    valid_embeddings.append(self.w2v_model[kmer])
            
            if valid_embeddings:
                # Average the embeddings
                embeddings[i] = np.mean(valid_embeddings, axis=0)
        
        return embeddings
        
    def extract_bow_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Extract Bag-of-Words k-mer embeddings."""
        logger.info("Generating Bag-of-Words k-mer embeddings...")
        vocab_kmers = set()
        for s in sequences:
            kmers = self._extract_kmers_from_sequence(s)
            vocab_kmers.update(kmers)
        
        vocab = {k: i for i, k in enumerate(sorted(list(vocab_kmers)))}
        if not vocab:
            logger.warning("K-mer vocabulary is empty. Returning zero vectors for embeddings.")
            return np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)

        vocab_size = len(vocab)
        logger.info(f"Created k-mer vocabulary of size {vocab_size}.")
        embeddings = np.zeros((len(sequences), vocab_size), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            kmers = self._extract_kmers_from_sequence(seq)
            if kmers:
                indices = [vocab[kmer] for kmer in kmers if kmer in vocab]
                if indices:
                    counts = np.bincount(indices, minlength=vocab_size)
                    embeddings[i] = counts / len(indices)

        if vocab_size > self.embedding_dim:
            logger.info(f"Vocabulary size ({vocab_size}) > embedding dim ({self.embedding_dim}). "
                       "Applying Incremental PCA for dimensionality reduction.")
            return IncrementalPCA(n_components=self.embedding_dim, 
                                batch_size=min(512, len(sequences))).fit_transform(embeddings)
        elif vocab_size < self.embedding_dim:
            logger.info(f"Vocabulary size ({vocab_size}) < embedding dim ({self.embedding_dim}). "
                       "Padding with zeros.")
            return np.hstack([embeddings, np.zeros((len(sequences), self.embedding_dim - vocab_size))])
        
        return embeddings
    
    def extract_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Extract embeddings using the specified method."""
        if self.embedding_method == "w2v":
            return self.extract_w2v_embeddings(sequences)
        elif self.embedding_method == "bow":
            return self.extract_bow_embeddings(sequences)
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")

class StatisticalAnalyzer:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        logger.info(f"Initialized StatisticalAnalyzer with alpha={self.alpha}")
        
    def _dunn_posthoc_test(self, data, class_labels, feature_name):
        df = pd.DataFrame({'value': data, 'class': class_labels})
        try:
            p_values = sp.posthoc_dunn(df, val_col='value', group_col='class', p_adjust='fdr_bh')
            return p_values.to_dict()
        except Exception as e:
            logger.warning(f"Could not perform Dunn's post-hoc test for feature '{feature_name}': {e}")
            return None
            
    def univariate_feature_analysis(self, X, y, names, encoder):
        logger.info("Performing univariate feature analysis...")
        results = {}
        unique_labels, unique_codes = encoder.classes_, encoder.transform(encoder.classes_)
        
        for i, name in enumerate(tqdm(names, desc="Univariate Tests", leave=False)):
            data = X[:, i]
            res = {
                "mean": np.mean(data), 
                "std": np.std(data), 
                "class_stats": {
                    c_name: {
                        "mean": np.mean(data[y==c_code]), 
                        "median": np.median(data[y==c_code])
                    } 
                    for c_name, c_code in zip(unique_labels, unique_codes)
                }
            }
            
            by_class = [data[y == c] for c in unique_codes if np.sum(y == c) > 0]
            if len(by_class) < 2: 
                continue
            
            p_val, effect_size = 1.0, 0.0
            try:
                if len(unique_labels) == 2:
                    _, p_val = stats.mannwhitneyu(by_class[0], by_class[1], alternative='two-sided')
                    if np.std(data) > 0: 
                        effect_size = (np.mean(by_class[0])-np.mean(by_class[1]))/np.std(data)
                else:
                    _, p_val = stats.kruskal(*by_class)
                    if len(data) > len(unique_labels): 
                        effect_size = (len(data) - len(unique_labels)) / len(data)
            except ValueError as e:
                if 'All numbers are identical' in str(e):
                    logger.debug(f"Skipping identical-value feature '{name}' for statistical test.")
                else: 
                    logger.error(f"Error in statistical test for feature '{name}': {e}")

            res["test"] = {"p_value": p_val, "effect_size": effect_size}
            if len(unique_labels) > 2 and p_val < self.alpha:
                res["posthoc"] = self._dunn_posthoc_test(data, encoder.inverse_transform(y), name)
            results[name] = res
        return results
        
    def multiple_testing_correction(self, results):
        logger.info("Applying multiple testing correction (FDR-BH).")
        p_vals_list = [(name, r['test']['p_value']) for name, r in results.items() 
                       if 'test' in r and 'p_value' in r['test']]
        if not p_vals_list:
            logger.warning("No p-values found to correct.")
            return results
            
        names, p_vals = zip(*p_vals_list)
        try:
            _, p_corr, _, _ = multipletests(np.nan_to_num(p_vals, nan=1.0), 
                                         method="fdr_bh", alpha=self.alpha)
            for i, name in enumerate(names): 
                results[name]["test"]["p_corr"] = p_corr[i]
        except Exception as e:
            logger.error(f"Failed to perform multiple testing correction: {e}")
        return results
        
    def feature_interaction_analysis(self, X, y, names, max_pairs=2000):
        logger.info("Analyzing feature interactions using mutual information.")
        X_s = StandardScaler().fit_transform(X)
        num_top_features = min(50, X.shape[1])
        f_scores, _ = f_classif(X_s, y)
        top_indices = np.argsort(f_scores)[-num_top_features:]
        
        X_sub, names_sub = X_s[:, top_indices], [names[i] for i in top_indices]
        all_pairs = list(itertools.combinations(range(len(names_sub)), 2))
        
        if len(all_pairs) == 0:
            logger.warning("Not enough features to analyze interactions.")
            return {"top_interactions": []}

        pairs_to_sample = min(max_pairs, len(all_pairs))
        pairs = random.sample(all_pairs, pairs_to_sample)
        interactions = []
        
        for i, j in tqdm(pairs, desc="Interaction Tests", leave=False):
            mi_i = mutual_info_classif(X_sub[:,i:i+1], y, random_state=42)[0]
            mi_j = mutual_info_classif(X_sub[:,j:j+1], y, random_state=42)[0]
            mi_inter = mutual_info_classif((X_sub[:,i]*X_sub[:,j]).reshape(-1,1), y, random_state=42)[0]
            interactions.append({
                "f1": names_sub[i], 
                "f2": names_sub[j], 
                "synergy": mi_inter - (mi_i + mi_j)
            })
        
        logger.info(f"Found {len(interactions)} feature interactions.")
        return {"top_interactions": sorted(interactions, key=lambda x: abs(x["synergy"]), reverse=True)[:20]}
        
    def permutation_importance_test(self, model, X_test, y_test, names):
        logger.info("Calculating feature importance using permutation on the test set.")
        base_score = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        logger.info(f"Base model AUC on test set: {base_score:.4f}")
        results = {}
        
        for i, name in enumerate(tqdm(names, desc="Permutation Importance", leave=False)):
            X_p = X_test.copy()
            np.random.shuffle(X_p[:, i])
            permuted_score = roc_auc_score(y_test, model.predict_proba(X_p), multi_class='ovr')
            results[name] = {"importance_mean": base_score - permuted_score}
        return results
        
    def comprehensive_statistical_analysis(self, X_train, y_train, X_test, y_test, names, model, encoder):
        logger.info("Starting comprehensive statistical analysis...")
        univariate = self.univariate_feature_analysis(X_train, y_train, names, encoder)
        corrected = self.multiple_testing_correction(univariate)
        interactions = self.feature_interaction_analysis(X_train, y_train, names)
        perms = self.permutation_importance_test(model, X_test, y_test, names)
        logger.info("Comprehensive statistical analysis complete.")
        return {
            "univariate": corrected, 
            "interactions": interactions, 
            "permutation": perms
        }

class AnalysisVisualizer:
    def __init__(self, outdir: Path, encoder: LabelEncoder, n_jobs: int = -1):
        self.outdir = outdir
        self.encoder = encoder
        self.n_jobs = n_jobs
        logger.info(f"Initialized AnalysisVisualizer, output directory: {self.outdir}")

    def plot_class_balance(self, labels: List[str]):
        logger.info("Plotting class balance.")
        plt.figure(figsize=(10, 6))
        sns.countplot(y=labels, order=pd.Series(labels).value_counts().index)
        plt.title("Class Distribution")
        plt.xlabel("Number of Samples")
        plt.ylabel("Class")
        plt.tight_layout()
        plt.savefig(self.outdir / "class_balance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves_multiclass(self, results: Dict[str, Dict]):
        logger.info("Plotting ROC curves...")
        best_model_name = max(results, key=lambda k: results[k]['auc'])
        res = results[best_model_name]
        y_test, y_prob = res['y_test'], res['y_prob']
        n_classes = len(self.encoder.classes_)

        plt.figure(figsize=(12, 10))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{best_model_name} (AUC = {roc_auc:.3f})")
        else:
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                        label=f'ROC for {self.encoder.classes_[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Chance")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - {best_model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(self.outdir / "roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pr_curves(self, results: Dict[str, Dict]):
        logger.info("Plotting Precision-Recall curves...")
        plt.figure(figsize=(12, 10))
        for name, res in results.items():
            if len(self.encoder.classes_) == 2:
                precision, recall, _ = precision_recall_curve(res['y_test'], res['y_prob'][:, 1])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2, label=f"{name} (AUC = {pr_auc:.3f})")
        
        if results:
            res = next(iter(results.values()))
            no_skill = np.sum(res['y_test']==1) / len(res['y_test'])
            plt.plot([0, 1], [no_skill, no_skill], 'k--', 
                    label=f'No Skill (baseline={no_skill:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Binary Classification')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.outdir / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        logger.info(f"Plotting confusion matrix for model: {model_name}")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.encoder.classes_, 
                   yticklabels=self.encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(self.outdir / f"confusion_matrix_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_performance(self, results: Dict[str, Dict]):
        logger.info("Plotting overall model performance summary...")
        df = pd.DataFrame({
            k: {"CV AUC": v['cv_auc'], "Test AUC": v['auc']} 
            for k, v in results.items()
        }).T.sort_values("Test AUC", ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        df.plot(kind='bar', ax=ax1, title="Model Performance (AUC Scores)", rot=45)
        ax1.set_ylabel("ROC AUC Score")
        ax1.grid(axis='y', linestyle='--')
        
        report_data = {model: res['report']['macro avg'] 
                      for model, res in results.items()}
        report_df = pd.DataFrame(report_data).T[['precision', 'recall', 'f1-score']]
        sns.heatmap(report_df, annot=True, cmap='viridis', ax=ax2)
        ax2.set_title("Classification Metrics (Macro Avg)")
        
        plt.tight_layout()
        plt.savefig(self.outdir / "model_performance_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_dim_reduction(self, data: Dict[str, np.ndarray], y: np.ndarray):
        logger.info("Plotting dimensionality reduction results (PCA, t-SNE, UMAP)...")
        valid_data = {name: d for name, d in data.items() if d is not None}
        num_plots = len(valid_data)
        
        if num_plots == 0:
            logger.warning("No dimensionality reduction data to plot.")
            return
            
        fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), squeeze=False)
        axes = axes.flatten()
        titles = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
        
        plot_idx = 0
        for name, d in valid_data.items():
            ax = axes[plot_idx]
            for j, label_code in enumerate(np.unique(y)):
                mask = y == label_code
                ax.scatter(d[mask, 0], d[mask, 1], 
                          label=self.encoder.classes_[j], alpha=0.7, s=10)
            ax.set_title(titles.get(name, name))
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            plot_idx += 1
        
        if plot_idx > 0:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', title="Classes")
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(self.outdir / "dimensionality_reduction.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_correlation_heatmap(self, X_df: pd.DataFrame, top_feature_names: List[str]):
        logger.info(f"Plotting feature correlation heatmap for top {len(top_feature_names)} features.")
        if len(top_feature_names) < 2:
            logger.warning("Not enough features to plot a correlation heatmap.")
            return
        
        corr_matrix = X_df[top_feature_names].corr()
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
        plt.title(f'Feature Correlation Matrix (Top {len(top_feature_names)} Features)')
        plt.tight_layout()
        plt.savefig(self.outdir / "feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_top_feature_distributions(self, X_df: pd.DataFrame, y: np.ndarray, top_feature_names: List[str]):
        logger.info(f"Plotting distributions for top {len(top_feature_names)} features.")
        df = X_df[top_feature_names].copy()
        df['class'] = self.encoder.inverse_transform(y)
        
        num_features = len(top_feature_names)
        n_cols = 2
        n_rows = (num_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if num_features > 1 else [axes]
        axes = axes.flatten()
        
        for i, feature in enumerate(top_feature_names):
            sns.violinplot(data=df, x='class', y=feature, ax=axes[i], inner='quartile', cut=0)
            axes[i].set_title(f'Distribution of {feature} by Class')
            axes[i].tick_params(axis='x', rotation=30)
        
        for i in range(num_features, len(axes)): 
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.savefig(self.outdir / "top_feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_feature_analysis_dashboard(self, stats: Dict, fi: Dict, feature_names: List[str]):
        logger.info("Creating feature analysis dashboard...")
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle("Feature Analysis Dashboard", fontsize=24, y=0.98)
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i, j]) for i, j in itertools.product(range(2), range(2))]
        
        # Model importance plot
        if fi:
            best_models = [k for k, v in fi.items() if v is not None]
            if best_models:
                best_model = max(best_models, key=lambda k: np.mean(fi[k]))
                imp_df = pd.DataFrame({"imp": fi[best_model]}, index=feature_names)
                imp_df.nlargest(20, 'imp').plot(kind='barh', ax=ax1, 
                                               title=f"Top Features by Model Importance ({best_model})", 
                                               legend=False)
                ax1.invert_yaxis()
        
        # Permutation importance plot
        if stats.get('permutation'):
            perm_imp = pd.DataFrame.from_dict(stats['permutation'], orient='index')
            if not perm_imp.empty:
                perm_imp.nlargest(20, 'importance_mean')['importance_mean'].plot(
                    kind='barh', ax=ax2, title="Top Features by Permutation Importance")
                ax2.invert_yaxis()

        # Univariate significance plot
        if stats.get('univariate'):
            p_corr_data = {k: v['test'].get('p_corr', 1.0) 
                          for k, v in stats['univariate'].items() if 'test' in v}
            if p_corr_data:
                top_univariate = pd.Series(p_corr_data).nsmallest(20)
                (-np.log10(top_univariate.replace(0, 1e-300))).plot(
                    kind='barh', ax=ax3, title="Top Significant Features (-log10 adj. p-value)", 
                    color='coral')
                ax3.invert_yaxis()

        # Feature interactions plot
        if stats.get('interactions') and stats['interactions']['top_interactions']:
            top_interactions = pd.DataFrame(stats['interactions']['top_interactions'])
            top_interactions['label'] = (top_interactions['f1'].str[:15] + ' & ' + 
                                        top_interactions['f2'].str[:15])
            top_interactions.set_index('label')['synergy'].plot(
                kind='barh', ax=ax4, title="Top Feature Interactions by Synergy", color='teal')
            ax4.invert_yaxis()
        else:
            ax4.text(0.5, 0.5, "No significant interactions found.", 
                    ha='center', va='center', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.outdir / "feature_analysis_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_calibration_curve(self, model: Any, model_name: str, X_test: np.ndarray, y_test: np.ndarray):
        logger.info(f"Plotting calibration curve(s) for {model_name}.")
        n_classes = len(self.encoder.classes_)

        if n_classes <= 2:
            fig, ax = plt.subplots(figsize=(10, 10))
            CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10, ax=ax, name=model_name)
            ax.set_title(f'Calibration Curve for {model_name}')
        else:
            n_rows = (n_classes + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1) if n_classes > 1 else [axes]
            axes = axes.flatten()

            for i in range(n_classes):
                ax = axes[i]
                y_test_ovr = (y_test == i).astype(int)
                y_prob_ovr = model.predict_proba(X_test)[:, i]
                
                CalibrationDisplay.from_predictions(
                    y_test_ovr, y_prob_ovr, n_bins=10, ax=ax, 
                    name=f'Class: {self.encoder.classes_[i]}'
                )
                ax.set_title(f'Calibration for Class: {self.encoder.classes_[i]}')
            
            for i in range(n_classes, len(axes)): 
                axes[i].set_visible(False)
            fig.suptitle(f'One-vs-Rest Calibration Curves for {model_name}', fontsize=16)

        plt.tight_layout()
        plt.savefig(self.outdir / f"calibration_curves_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_learning_curves(self, model: Any, model_name: str, X: np.ndarray, y: np.ndarray):
        logger.info(f"Generating learning curves for {model_name}...")
        
        min_class_count = np.min(np.bincount(y))
        n_splits = min(3, min_class_count)

        if n_splits < 2:
            logger.warning(f"Skipping learning curves: smallest class has only {min_class_count} members.")
            return

        try:
            cv_strat = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=cv_strat, n_jobs=self.n_jobs,
                train_sizes=np.linspace(.1, 1.0, 5), scoring="roc_auc_ovr",
                error_score='raise'
            )
            train_scores_mean, train_scores_std = np.nanmean(train_scores, axis=1), np.nanstd(train_scores, axis=1)
            test_scores_mean, test_scores_std = np.nanmean(test_scores, axis=1), np.nanstd(test_scores, axis=1)

            plt.figure(figsize=(10, 6))
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                           train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                           test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            plt.xlabel("Training examples")
            plt.ylabel("ROC AUC OVR Score")
            plt.title(f"Learning Curves ({model_name})")
            plt.legend(loc="best")
            plt.grid(True)
            plt.savefig(self.outdir / f"learning_curve_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Could not generate learning curves for {model_name}: {e}")

    def plot_feature_importance_comparison(self, fi_dict: Dict, feature_names: List[str], top_n: int = 20):
        logger.info("Plotting feature importance comparison across models.")
        valid_fi = {k: v for k, v in fi_dict.items() if v is not None}
        if not valid_fi:
            logger.warning("No feature importances available to compare.")
            return
            
        fi_df = pd.DataFrame(valid_fi, index=feature_names).dropna(axis=1, how='all')
        if fi_df.empty:
            logger.warning("No valid feature importances to compare.")
            return

        fi_df['mean_imp'] = fi_df.mean(axis=1)
        top_features = fi_df.nlargest(top_n, 'mean_imp')
        
        plot_df = top_features.drop('mean_imp', axis=1)
        if not plot_df.empty:
            plot_df.plot(kind='barh', figsize=(12, max(8, len(plot_df) * 0.4)),
                        width=0.8, colormap='viridis')
            plt.title(f'Top {len(top_features)} Feature Importances Across Models')
            plt.xlabel("Importance Score")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.outdir / "feature_importance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

    def plot_shap_summary(self, model: Any, model_name: str, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame):
        logger.info(f"Generating SHAP summary plot for {model_name}...")
        try:
            explainer = shap.Explainer(model, X_train_df)
            shap_values = explainer(X_test_df)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_df, show=False, class_names=self.encoder.classes_)
            plt.title(f"SHAP Feature Importance Summary ({model_name})")
            plt.savefig(self.outdir / f"shap_summary_plot_{model_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Could not generate SHAP summary plot: {e}")


class EnhancedDNAStructurePredictor:
    def __init__(self, n_jobs=-1, embedding_config: Optional[Dict] = None):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        logger.info(f"Initializing EnhancedDNAStructurePredictor with n_jobs={self.n_jobs}")
        
        self.feature_extractor = AdvancedDNAFeatureExtractor()
        
        # Initialize embedding extractor with config
        embedding_config = embedding_config or {}
        self.embedding_extractor = DNAEmbeddingExtractor(**embedding_config)
        
        self.statistical_analyzer = StatisticalAnalyzer()
        self.models, self.scalers, self.label_encoder = {}, {}, LabelEncoder()
        self.feature_names = []
        self._initialize_models()

    def _initialize_models(self):
        logger.info("Initializing classification models.")
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=self.n_jobs),
            "lightgbm": lgb.LGBMClassifier(
                random_state=42, n_jobs=self.n_jobs, verbose=-1),
            "xgboost": xgb.XGBClassifier(
                random_state=42, n_jobs=self.n_jobs, eval_metric="mlogloss", 
                use_label_encoder=False, tree_method='hist'),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "logistic_regression": LogisticRegression(
                random_state=42, max_iter=1000, n_jobs=self.n_jobs),
            "svm": SVC(probability=True, random_state=42, class_weight='balanced'),
        }

    def _open_compressed(self, filepath: Union[str, Path], mode: str = 'rt'):
        filepath = Path(filepath)
        if filepath.suffix == '.gz': 
            return gzip.open(filepath, mode)
        return open(filepath, mode)

    def load_sequences_from_fasta(self, path: str, stype: str) -> StructureData:
        logger.info(f"Loading FASTA for '{stype}' from: {path}")
        with self._open_compressed(path, 'rt') as handle:
            sequences = [str(r.seq) for r in SeqIO.parse(handle, "fasta")]
        logger.info(f"Loaded {len(sequences)} sequences for type '{stype}'.")
        return StructureData(sequences=sequences, labels=[stype] * len(sequences), structure_type=stype)
        
    def extract_features_parallel(self, sequences: List[str]) -> np.ndarray:
        logger.info(f"Extracting features for {len(sequences)} sequences using {self.n_jobs} processes...")
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            handcrafted = list(tqdm(
                executor.map(self.feature_extractor.extract_all_features, sequences), 
                total=len(sequences), desc="Handcrafted Features"))
        
        handcrafted_df = pd.DataFrame(handcrafted).fillna(0)
        self.feature_names = handcrafted_df.columns.tolist()
        logger.info(f"Generated {len(self.feature_names)} handcrafted features.")
        
        embeds = self.embedding_extractor.extract_embeddings(sequences)
        logger.info(f"Generated {embeds.shape[1]} embedding features.")
        self.feature_names.extend([f"embed_{i}" for i in range(embeds.shape[1])])
        
        X = np.hstack([handcrafted_df.values.astype(np.float32), embeds])
        logger.info(f"Final feature matrix shape: {X.shape}")
        return X

    def train_models(self, X: np.ndarray, y_enc: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info(f"Splitting data ({X.shape[0]} samples, {X.shape[1]} features) into training and testing sets.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        self.scalers['combined'] = scaler
        logger.info(f"Train set: {X_train_s.shape}, Test set: {X_test_s.shape}")

        min_class_count_train = np.min(np.bincount(y_train))
        n_splits = min(3, min_class_count_train)
        
        if n_splits < 2:
            logger.error(f"Cannot perform cross-validation: smallest class has only {min_class_count_train} members.")
            n_splits = 1

        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_s, y_train)
            
            cv_mean = np.nan
            if n_splits > 1:
                logger.info(f"  Performing {n_splits}-fold cross-validation for {name}...")
                cv_jobs = 1 if name == 'svm' else self.n_jobs
                cv_strat = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                try:
                    cv = cross_val_score(model, X_train_s, y_train, cv=cv_strat, 
                                       scoring='roc_auc_ovr', n_jobs=cv_jobs, error_score='raise')
                    cv_mean = np.nanmean(cv)
                except Exception as e:
                    logger.error(f"CV for {name} failed: {e}")

            logger.info(f"  Evaluating {name} on the test set...")
            y_prob = model.predict_proba(X_test_s)
            y_pred = model.predict(X_test_s)
            
            try: 
                auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except ValueError as e:
                logger.warning(f"Could not calculate ROC AUC for {name}: {e}")
                auc_score = 0.0

            report = classification_report(y_test, y_pred, 
                                         target_names=self.label_encoder.classes_, 
                                         output_dict=True)
            results[name] = {
                "model": model, "cv_auc": cv_mean, "auc": auc_score, 
                "report": report, "y_test": y_test, "y_prob": y_prob, "y_pred": y_pred
            }
            logger.info(f"  {name} -> Test AUC: {auc_score:.4f}, CV AUC: {cv_mean:.4f}")
        
        return results, X_train_s, y_train, X_test_s, y_test

    def analyze_feature_importance(self, results: Dict) -> Dict:
        logger.info("Extracting feature importances from trained models.")
        fi = {}
        for name, res in results.items():
            model = res["model"]
            if hasattr(model, "feature_importances_"):
                fi[name] = model.feature_importances_
            elif hasattr(model, "coef_"):
                coef = model.coef_
                fi[name] = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
            else: 
                fi[name] = None
        return fi

    def perform_dimensionality_reduction(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Optional[np.ndarray]], np.ndarray]:
        logger.info("Performing dimensionality reduction for visualization...")
        
        subset_size = min(5000, X.shape[0])
        if X.shape[0] > subset_size:
            logger.info(f"Subsampling to {subset_size} points for dimensionality reduction plots.")
            indices = np.random.choice(X.shape[0], subset_size, replace=False)
            X_sub, y_sub = X[indices], y[indices]
        else: 
            X_sub, y_sub = X, y

        X_s_sub = self.scalers['combined'].transform(X_sub)
        
        if X_s_sub.shape[0] < 2:
            logger.warning("Not enough samples for dimensionality reduction.")
            return {"pca": None, "tsne": None, "umap": None}, y_sub
        
        pca_plot_data, tsne_data, umap_data = None, None, None
        
        # PCA
        try:
            num_pca_components = min(2, X_s_sub.shape[1], X_s_sub.shape[0] - 1)
            if num_pca_components >= 2:
                pca_plot_data = PCA(n_components=2, random_state=42).fit_transform(X_s_sub)
        except Exception as e: 
            logger.error(f"PCA failed: {e}")

        # PCA for manifold learning input
        pca_for_manifold = None
        num_pca_manifold_components = min(50, X_s_sub.shape[1], X_s_sub.shape[0] - 1)
        if num_pca_manifold_components >= 2:
             pca_for_manifold = PCA(n_components=num_pca_manifold_components, 
                                   random_state=42).fit_transform(X_s_sub)

        if pca_for_manifold is not None:
            # t-SNE
            try:
                tsne_perplexity = min(30, X_s_sub.shape[0] - 1)
                tsne_data = TSNE(n_components=2, perplexity=tsne_perplexity, 
                               random_state=42, n_jobs=self.n_jobs).fit_transform(pca_for_manifold)
            except Exception as e: 
                logger.error(f"t-SNE failed: {e}")
            
            # UMAP
            try:
                umap_data = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, 
                                    random_state=42).fit_transform(pca_for_manifold)
            except Exception as e: 
                logger.error(f"UMAP failed: {e}")
        
        return {"pca": pca_plot_data, "tsne": tsne_data, "umap": umap_data}, y_sub

    def generate_report(self, results: Dict, stats: Dict, fi: Dict, outdir: Path):
        logger.info("Generating final text report...")
        with open(outdir / "analysis_report.txt", "w") as f:
            f.write("DNA STRUCTURE ANALYSIS REPORT\n" + "="*50 + "\n\n")
            f.write("1. MODEL PERFORMANCE\n" + "-"*50 + "\n")
            for name, res in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
                f.write(f"- {name:<20} Test AUC: {res['auc']:.4f}, CV AUC: {res['cv_auc']:.4f}\n")
            
            f.write("\n\n2. TOP 10 SIGNIFICANT FEATURES (UNIVARIATE ANALYSIS)\n" + "-"*50 + "\n")
            if stats.get('univariate'):
                top_feats = sorted(stats['univariate'].items(), 
                                 key=lambda x: x[1].get('test', {}).get('p_corr', 1))[:10]
                for name, res in top_feats:
                    p_corr = res.get('test', {}).get('p_corr', 1.0)
                    f.write(f"- {name:<40} adj. p-value = {p_corr:.2e}\n")

            if len(self.label_encoder.classes_) > 2 and stats.get('univariate'):
                f.write("\n\n3. TOP SIGNIFICANT PAIRWISE DIFFERENCES (POST-HOC)\n" + "-"*50 + "\n")
                posthoc_hits = []
                for feat, res in stats['univariate'].items():
                    if 'posthoc' in res and res['posthoc'] is not None:
                        try:
                            ph_df = pd.DataFrame(res['posthoc']).unstack().dropna()
                            ph_df = ph_df[ph_df.index.get_level_values(0) != ph_df.index.get_level_values(1)]
                            if not ph_df.empty:
                                best_pair = ph_df.idxmin()
                                posthoc_hits.append((feat, best_pair[0], best_pair[1], ph_df.min()))
                        except Exception:
                            continue
                            
                for feat, c1, c2, p_val in sorted(posthoc_hits, key=lambda x: x[3])[:10]:
                    f.write(f"- Feature '{feat}': Difference between '{c1}' and '{c2}' (p={p_val:.2e})\n")

            # Feature importance section
            valid_fi = {k: v for k, v in fi.items() if v is not None}
            if valid_fi:
                best_model_name = max(valid_fi, key=lambda k: np.mean(fi[k]))
                f.write(f"\n\n4. TOP 10 FEATURES BY IMPORTANCE (from {best_model_name})\n" + "-"*50 + "\n")
                df_imp = pd.DataFrame({"imp": fi[best_model_name]}, index=self.feature_names)
                for name, row in df_imp.nlargest(10, 'imp').iterrows(): 
                    f.write(f"- {name:<40} Importance: {row['imp']:.4f}\n")

    def run_analysis(self, data: Union[List[StructureData], Dict[str, str]], 
                    outdir: Union[str, Path], ref_genome: Optional[str] = None):
        outdir = Path(outdir)
        plot_dir = outdir / "plots"
        outdir.mkdir(exist_ok=True, parents=True)
        plot_dir.mkdir(exist_ok=True)
        logger.info(f"Starting analysis pipeline. Output will be saved to: {outdir}")
        
        logger.info("--- Step 1: Loading and Processing Sequence Data ---")
        if isinstance(data, dict):
            if ref_genome is None:
                raise ValueError("ref_genome must be provided when using BED files")
            processor = BedFileProcessor(ref_genome)
            processed_beds = processor.merge_and_deduplicate_beds(data)
            data_list = processor.extract_sequences_from_beds(processed_beds)
        else:
            data_list = data
            
        sequences = [s for d in data_list for s in d.sequences]
        labels = [l for d in data_list for l in d.labels]
        
        if len(np.unique(labels)) < 2: 
            raise ValueError(f"Need at least two classes. Found: {np.unique(labels)}")
        
        class_counts = Counter(labels)
        logger.info(f"Data loaded. Total sequences: {len(sequences)}. Class distribution: {class_counts}")
        
        if any(c < 2 for c in class_counts.values()): 
            raise ValueError(f"All classes need at least 2 members. Found: {class_counts}")
        elif any(c < 20 for c in class_counts.values()): 
            logger.warning(f"Some classes have fewer than 20 samples: {class_counts}")

        y_encoded = self.label_encoder.fit_transform(labels)
        self.visualizer = AnalysisVisualizer(plot_dir, self.label_encoder, self.n_jobs)

        logger.info("--- Step 2: Feature Extraction ---")
        X = self.extract_features_parallel(sequences)
        
        logger.info("--- Step 3: Model Training and Evaluation ---")
        model_results, X_train, y_train, X_test, y_test = self.train_models(X, y_encoded)
        best_model_name = max(model_results, key=lambda k: model_results[k]['auc'])
        best_model_res = model_results[best_model_name]
        best_model = best_model_res['model']
        logger.info(f"Best model: '{best_model_name}' with Test AUC: {best_model_res['auc']:.4f}")
        
        X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
        X_test_df = pd.DataFrame(X_test, columns=self.feature_names)
        
        logger.info("--- Step 4: Statistical and Feature Analysis ---")
        stat_results = self.statistical_analyzer.comprehensive_statistical_analysis(
            X_train, y_train, X_test, y_test, self.feature_names, best_model, self.label_encoder)
        feature_importances = self.analyze_feature_importance(model_results)
        
        logger.info("--- Step 5: Generating All Plots and Reports ---")
        self.visualizer.plot_class_balance(labels)
        self.visualizer.plot_model_performance(model_results)
        self.visualizer.plot_roc_curves_multiclass(model_results)
        
        if len(self.label_encoder.classes_) == 2: 
            self.visualizer.plot_pr_curves(model_results)
            
        self.visualizer.plot_confusion_matrix(y_test, best_model_res['y_pred'], best_model_name)
        self.visualizer.plot_calibration_curve(best_model, best_model_name, X_test, y_test)
        self.visualizer.plot_learning_curves(best_model, best_model_name, 
                                           self.scalers['combined'].transform(X), y_encoded)
        
        if stat_results.get('permutation'):
            top_perm_imp_names = (pd.DataFrame.from_dict(stat_results['permutation'], orient='index')
                                 .nlargest(10, 'importance_mean').index.tolist())
            top_corr_imp_names = (pd.DataFrame.from_dict(stat_results['permutation'], orient='index')
                                 .nlargest(25, 'importance_mean').index.tolist())
            
            self.visualizer.plot_top_feature_distributions(X_train_df, y_train, top_perm_imp_names)
            self.visualizer.plot_feature_correlation_heatmap(X_train_df, top_corr_imp_names)
        
        self.visualizer.plot_feature_importance_comparison(feature_importances, self.feature_names)
        self.visualizer.create_feature_analysis_dashboard(stat_results, feature_importances, self.feature_names)
        self.visualizer.plot_shap_summary(best_model, best_model_name, X_train_df, X_test_df)
        
        dim_reduction_data, y_for_plotting = self.perform_dimensionality_reduction(X, y_encoded)
        self.visualizer.plot_dim_reduction(dim_reduction_data, y_for_plotting)
        self.generate_report(model_results, stat_results, feature_importances, outdir)
        
        logger.info("--- Step 6: Saving Analysis Bundle ---")
        bundle_path = outdir / "analysis_bundle.pkl"
        bundle_data = {
            "models": {k: v['model'] for k, v in model_results.items()}, 
            "stats": stat_results, 
            "fi": feature_importances, 
            "scaler": self.scalers['combined'], 
            "encoder": self.label_encoder, 
            "feature_names": self.feature_names
        }
        
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle_data, f)
        
        logger.info(f"Analysis complete. All assets saved to {outdir}. Bundle saved to {bundle_path}")
        
        return {"models": model_results, "stats": stat_results}


def create_argument_parser():
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced DNA Structure Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with FASTA files
  python script.py --input data/ --output results/

  # Analysis with Word2Vec embeddings
  python script.py --input data/ --output results/ --embedding-method w2v --w2v-model dna2vec.model

  # Analysis with BED files and reference genome
  python script.py --input beds.json --output results/ --ref-genome hg38.fa --input-type bed

  # Custom parameters
  python script.py --input data/ --output results/ --kmer-size 8 --embedding-dim 200 --n-jobs 4

  # Exclude specific models
  python script.py --input data/ --output results/ --exclude-models svm xgboost

  # Generate sample data and run analysis
  python script.py --generate-sample --sample-size 500 --output results/
        """)
    
    # Input/Output arguments
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', '-i', type=str,
                         help='Input directory containing FASTA files or JSON file with BED file paths')
    io_group.add_argument('--output', '-o', type=str, required=True,
                         help='Output directory for results')
    io_group.add_argument('--input-type', choices=['fasta', 'bed'], default='fasta',
                         help='Type of input data (default: fasta)')
    io_group.add_argument('--ref-genome', type=str,
                         help='Reference genome file (required for BED input)')
    
    # Embedding arguments
    embed_group = parser.add_argument_group('Embedding Configuration')
    embed_group.add_argument('--embedding-method', choices=['bow', 'w2v'], default='bow',
                            help='Embedding method: bow (Bag-of-Words) or w2v (Word2Vec) (default: bow)')
    embed_group.add_argument('--w2v-model', type=str,
                            help='Path to Word2Vec model file (.model, .vec, .bin)')
    embed_group.add_argument('--kmer-size', type=int, default=6,
                            help='K-mer size for embeddings (default: 6)')
    embed_group.add_argument('--embedding-dim', type=int, default=100,
                            help='Embedding dimension (default: 100, ignored for w2v)')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--exclude-models', nargs='+', 
                            choices=['random_forest', 'lightgbm', 'xgboost', 'gradient_boosting', 
                                   'logistic_regression', 'svm'],
                            help='Models to exclude from analysis')
    model_group.add_argument('--n-jobs', type=int, default=-1,
                            help='Number of parallel jobs (-1 for all cores, default: -1)')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--test-size', type=float, default=0.2,
                               help='Fraction of data for testing (default: 0.2)')
    analysis_group.add_argument('--cv-folds', type=int, default=3,
                               help='Number of cross-validation folds (default: 3)')
    analysis_group.add_argument('--random-seed', type=int, default=42,
                               help='Random seed for reproducibility (default: 42)')
    analysis_group.add_argument('--alpha', type=float, default=0.05,
                               help='Significance level for statistical tests (default: 0.05)')
    
    # Sample data generation
    sample_group = parser.add_argument_group('Sample Data Generation')
    sample_group.add_argument('--generate-sample', action='store_true',
                             help='Generate sample FASTA data for testing')
    sample_group.add_argument('--sample-size', type=int, default=100,
                             help='Number of sequences per class for sample data (default: 100)')
    sample_group.add_argument('--sample-classes', nargs='+', default=['gquad', 'hairpins', 'control'],
                             help='Class names for sample data (default: gquad hairpins control)')
    
    # Logging and output
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                           default='INFO', help='Logging level (default: INFO)')
    misc_group.add_argument('--no-plots', action='store_true',
                           help='Skip plot generation (faster for large datasets)')
    misc_group.add_argument('--config', type=str,
                           help='JSON configuration file (overrides command line arguments)')
    misc_group.add_argument('--save-config', type=str,
                           help='Save current configuration to JSON file')
    
    return parser


def load_config_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        raise ValueError(f"Failed to load configuration file {config_path}: {e}")


def save_config_file(args: argparse.Namespace, config_path: str):
    """Save configuration to JSON file."""
    config = {k: v for k, v in vars(args).items() if v is not None}
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")


def generate_sample_data(output_dir: Path, sample_size: int, class_names: List[str]):
    """Generate sample FASTA data for testing."""
    data_dir = output_dir / "sample_data"
    data_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Generating sample data with {sample_size} sequences per class")
    
    # GC content biases for different classes
    gc_biases = {
        'gquad': 0.7,
        'hairpins': 0.5, 
        'control': 0.4,
        'at_rich': 0.2,
        'gc_rich': 0.8
    }
    
    for class_name in class_names:
        gc_bias = gc_biases.get(class_name, 0.5)
        fasta_file = data_dir / f"{class_name}.fasta"
        
        with open(fasta_file, "w") as f:
            for i in range(sample_size):
                seq_length = random.randint(150, 300)
                weights = [(1-gc_bias)/2, (1-gc_bias)/2, gc_bias/2, gc_bias/2]
                seq = ''.join(random.choices("ATGC", k=seq_length, weights=weights))
                f.write(f">{class_name}_{i}\n{seq}\n")
        
        logger.info(f"Created {fasta_file} with {sample_size} sequences")
    
    return data_dir


def validate_arguments(args: argparse.Namespace):
    """Validate command line arguments."""
    if not args.generate_sample and not args.input:
        raise ValueError("Either --input or --generate-sample must be specified")
    
    if args.input_type == 'bed' and not args.ref_genome:
        raise ValueError("--ref-genome is required when using BED input")
    
    if args.embedding_method == 'w2v':
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is required for Word2Vec embeddings. Install with: pip install gensim")
        if not args.w2v_model:
            raise ValueError("--w2v-model is required when using Word2Vec embeddings")
        if not Path(args.w2v_model).exists():
            raise FileNotFoundError(f"Word2Vec model file not found: {args.w2v_model}")
    
    if args.test_size <= 0 or args.test_size >= 1:
        raise ValueError("--test-size must be between 0 and 1")
    
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    
    if args.kmer_size < 1:
        raise ValueError("--kmer-size must be at least 1")
    
    if args.embedding_dim < 1:
        raise ValueError("--embedding-dim must be at least 1")

def main():
    """Main entry point for the command line interface."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration file if provided
    if args.config:
        config = load_config_file(args.config)
        # Override command line args with config values
        for key, value in config.items():
            if hasattr(args, key) and value is not None:
                setattr(args, key, value)
    
    # Set up logging
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "dna_structure_analysis.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Save configuration if requested
    if args.save_config:
        save_config_file(args, args.save_config)
    
    try:
        # Validate arguments
        validate_arguments(args)
        
        logger.info("Starting Enhanced DNA Structure Analysis")
        logger.info(f"Command line arguments: {vars(args)}")
        
        # Generate sample data if requested
        if args.generate_sample:
            data_dir = generate_sample_data(output_dir, args.sample_size, args.sample_classes)
            if not args.input:
                args.input = str(data_dir)
                args.input_type = 'fasta'
        
        # Prepare embedding configuration
        embedding_config = {
            'embedding_dim': args.embedding_dim,
            'kmer_size': args.kmer_size,
            'embedding_method': args.embedding_method
        }
        
        if args.embedding_method == 'w2v':
            embedding_config['w2v_model_path'] = args.w2v_model
        
        # Initialize predictor
        predictor = EnhancedDNAStructurePredictor(
            n_jobs=args.n_jobs,
            embedding_config=embedding_config
        )
        
        # Remove excluded models
        if args.exclude_models:
            for model_name in args.exclude_models:
                if model_name in predictor.models:
                    del predictor.models[model_name]
                    logger.info(f"Excluded model: {model_name}")
        
        # Load input data
        if args.input_type == 'fasta':
            input_path = Path(args.input)
            if input_path.is_file():
                # Single FASTA file
                fasta_files = {input_path.stem.split('.')[0]: str(input_path)}
            else:
                # Directory of FASTA files
                fasta_files = {p.stem.split('.')[0]: str(p) 
                              for p in input_path.glob("*.fasta*")}
            
            if not fasta_files:
                raise FileNotFoundError(f"No FASTA files found in {args.input}")
            
            logger.info(f"Found FASTA files: {list(fasta_files.keys())}")
            data_list = [predictor.load_sequences_from_fasta(path, stype) 
                        for stype, path in fasta_files.items()]
            
        elif args.input_type == 'bed':
            # Load BED file configuration from JSON
            with open(args.input, 'r') as f:
                bed_files = json.load(f)
            
            logger.info(f"Found BED files: {list(bed_files.keys())}")
            data_list = bed_files  # Will be processed by run_analysis
        
        # Run analysis
        logger.info("Starting comprehensive analysis...")
        results = predictor.run_analysis(
            data=data_list,
            outdir=args.output,
            ref_genome=args.ref_genome if args.input_type == 'bed' else None
        )
        
        # Print summary
        print("\n" + "="*80)
        print("          DNA STRUCTURE ANALYSIS - FINAL SUMMARY")
        print("="*80)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir.resolve()}")
        
        print(f"\nEmbedding method: {args.embedding_method}")
        if args.embedding_method == 'w2v':
            print(f"Word2Vec model: {args.w2v_model}")
        print(f"K-mer size: {args.kmer_size}")
        print(f"Embedding dimension: {predictor.embedding_extractor.embedding_dim}")
        
        print(f"\nModels trained: {len(results['models'])}")
        active_models = [name for name in predictor.models.keys() 
                        if name not in (args.exclude_models or [])]
        print(f"Active models: {', '.join(active_models)}")
        
        print("\nModel Performance (Test Set AUC):")
        for name, res in sorted(results["models"].items(), 
                               key=lambda x: x[1]['auc'], reverse=True):
            cv_auc_str = f"{res['cv_auc']:.4f}" if not np.isnan(res['cv_auc']) else "N/A"
            print(f"  - {name:<25} Test AUC: {res['auc']:.4f}, CV AUC: {cv_auc_str}")
        
        if results['stats'].get('permutation'):
            print("\nTop 5 Most Important Features (by permutation):")
            top_perms = (pd.DataFrame.from_dict(results['stats']['permutation'], orient='index')
                        .nlargest(5, 'importance_mean'))
            for feature, row in top_perms.iterrows():
                print(f"  - {feature:<40} Importance Drop: {row['importance_mean']:.4f}")
        
        # Print file locations
        print(f"\nKey output files:")
        print(f"  - Analysis report: {output_dir / 'analysis_report.txt'}")
        print(f"  - Model bundle: {output_dir / 'analysis_bundle.pkl'}")
        print(f"  - Plots directory: {output_dir / 'plots'}")
        print(f"  - Log file: {output_dir / 'dna_structure_analysis.log'}")
        
        if args.generate_sample:
            print(f"  - Sample data: {output_dir / 'sample_data'}")
            
        print("\n" + "="*80)
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error in analysis pipeline: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Check the log file for detailed error information.")
        sys.exit(1)


def cli_main():
    """Entry point for command line usage."""
    main()


if __name__ == "__main__":
    # Support both direct execution and CLI usage
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        logger.info("No command line arguments provided. Running in interactive mode...")
        logger.info("For command line usage, run with --help for options")
        
        # Run the original main function for backward compatibility
        predictor = EnhancedDNAStructurePredictor(n_jobs=-1)
        
        data_dir = Path("data")
        if not data_dir.exists() or not any(data_dir.glob("*.fasta*")):
            logger.warning("Data directory 'data/' not found or empty. Creating sample FASTA files.")
            data_dir.mkdir(exist_ok=True)
            
            sample_configs = [
                ("gquad", 100, 0.7), 
                ("hairpins", 100, 0.5), 
                ("control", 100, 0.4)
            ]
            
            for stype, count, gc_bias in sample_configs:
                with open(data_dir / f"{stype}.fasta", "w") as f:
                    for i in range(count):
                        seq_length = random.randint(150, 250)
                        weights = [(1-gc_bias)/2, (1-gc_bias)/2, gc_bias/2, gc_bias/2]
                        seq = ''.join(random.choices("ATGC", k=seq_length, weights=weights))
                        f.write(f">{stype}_{i}\n{seq}\n")
            logger.info(f"Sample data created in '{data_dir.resolve()}'")

        fasta_files = {p.stem.split('.')[0]: str(p) for p in data_dir.glob("*.fasta*")}
        if not fasta_files: 
            logger.error("No FASTA files found in the 'data' directory.")
            sys.exit(1)

        logger.info(f"Found FASTA files: {list(fasta_files.keys())}")
        
        try:
            data_list = [predictor.load_sequences_from_fasta(path, stype) 
                        for stype, path in fasta_files.items()]
            results = predictor.run_analysis(data_list, "enhanced_fasta_analysis_results")
            
            print("\n" + "="*80)
            print("          DNA STRUCTURE ANALYSIS - FINAL SUMMARY")
            print("="*80)
            
            print("\nModel Performance (Test Set AUC):")
            for name, res in sorted(results["models"].items(), 
                                   key=lambda x: x[1]['auc'], reverse=True):
                print(f"  - {name:<25} Test AUC: {res['auc']:.4f}")
            
            print("\nTop 5 Most Important Features (by permutation):")
            if results['stats'].get('permutation'):
                top_perms = (pd.DataFrame.from_dict(results['stats']['permutation'], orient='index')
                            .nlargest(5, 'importance_mean'))
                for feature, row in top_perms.iterrows():
                    print(f"  - {feature:<40} Importance Drop: {row['importance_mean']:.4f}")
            else:
                print("  - No permutation importance data available")

        except Exception as e:
            logger.error(f"Critical error in analysis pipeline: {e}", exc_info=True)
    else:
        # Command line arguments provided, use CLI
        main()