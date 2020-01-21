"""
This provides utilities to compute the PARENT metric from Dhingra et al.
        Handling Divergent Reference Texts when Evaluating 
        Table-to-Text Generation. (Dhingra et al.) 2019

This code is slightly different from the original implementation, as
we don't compute the LCS score for overlap.

It is also specific to WikiBIO for now.

We provide code compatible with our version of ONMT, meaning you should
check the shape of the returned tensors from your model to be sure
everything works correctly.
"""

import collections
import itertools
import torch
import json
import onmt
import sys

import numpy as np

from onmt.utils.misc import sequence_mask
from onmt.utils.logging import logger


class Cleaner:
    """
    Base class for objects that handles model output and 
    returns source/target/gen as need for the metric.
    """
    def __init__(self, tgt_field):
        self.vocab = tgt_field.vocab
        self.eos_idx = self.vocab.stoi[tgt_field.eos_token]
        self.pad_idx = self.vocab.stoi[tgt_field.pad_token]
        self.bos_idx = self.vocab.stoi[tgt_field.init_token]
        self.unk_idx = self.vocab.stoi[tgt_field.unk_token]
        
    def clean_candidate_tokens(self, candidates, src_map, src_vocab, attns):
        """
        Builds the translation from model output.
        Replace <unk> by the input token with highest attn score
        """
        tokens = list()
        for idx, token in enumerate(candidates):
            token = token.item()
            if token == self.eos_idx:
                break

            if token == self.unk_idx and attns is not None:
                _, max_idx = attns[idx].max(0)
                max_idx = torch.nonzero(src_map[max_idx])
                clean_token = src_vocab.itos[max_idx.item()]        
            elif token < len(self.vocab):
                clean_token = self.vocab.itos[token] 
            else:
                clean_token = src_vocab.itos[token - len(self.vocab)]

            tokens.append(clean_token)
        return tokens, idx


class PARENT:
    def __init__(self, TABLE_VALUES, REF_NGRAM_COUNTS, REF_NGRAM_WEIGHTS):
        self.TABLE_VALUES = TABLE_VALUES
        self.REF_NGRAM_COUNTS = REF_NGRAM_COUNTS
        self.REF_NGRAM_WEIGHTS = REF_NGRAM_WEIGHTS
    
    @staticmethod
    def nwise(iterable, n):
        iterables = itertools.tee(iterable, n)
        [next(iterables[i]) for i in range(n) for j in range(i)]
        return zip(*iterables)
    
    def ngram_counts(self, sequence, order):
        """Returns count of all ngrams of given order in sequence."""
        if len(sequence) < order:
            return collections.Counter()
        return collections.Counter(self.nwise(sequence, order))
    
    @staticmethod
    def overlap_probability(ngram, table_values):
        return len(table_values.intersection(ngram)) / len(ngram)
    
    def __call__(self, prediction, bidx):
        
        table_values = self.TABLE_VALUES[bidx]
        ref_counts = self.REF_NGRAM_COUNTS[bidx]
        ref_weights = self.REF_NGRAM_WEIGHTS[bidx]
            
        # we set all precision and recall at zero and one by default
        order2precision = np.zeros(4)
        order2recall = np.zeros(4)

        for order in range(1, 5):
            pred_ngram_counts = self.ngram_counts(prediction, order)
            pred_ngram_weights = {ngram: self.overlap_probability(ngram, table_values)
                                  for ngram in pred_ngram_counts}

            # Compute precision via Equation (2)
            numerator = denominator = .0
            for ngram, count in pred_ngram_counts.items():
                denominator += count

                prob_ngram_in_ref = min(
                  1., ref_counts[order].get(ngram, 0) / count)

                numerator += count * (
                  prob_ngram_in_ref +
                  (1. - prob_ngram_in_ref) * pred_ngram_weights[ngram])

            if denominator != 0.:
                order2precision[order - 1] = numerator / denominator

            # Compute recall via Equation (5)
            numerator, denominator = 0., 0.
            for ngram, count in ref_counts[order].items():
                prob_ngram_in_pred = min(
                  1., pred_ngram_counts.get(ngram, 0) / count)
                denominator += count * ref_weights[order][ngram]
                numerator += count * ref_weights[order][ngram] * prob_ngram_in_pred

            if denominator != 0.:
                order2recall[order - 1] = numerator / denominator

        # Compute geometric averages of precision on all orders
        precision = 1e-5
        if order2precision.all():
            precision = np.exp(np.log(order2precision).sum() / 4)
            
        recall = 1e-5
        if order2recall.all():
            recall = np.exp(np.log(order2recall).sum() / 4)
            
        f1_score = 2 * (precision * recall) / (precision + recall)
                
        return f1_score
    
    
class PARENTLossCompute:
    def __init__(self, tgt_field, tv_path, rnc_path, rnw_path, ref_path):
        self.cleaner = Cleaner(tgt_field)
        
        logger.info("Initializing PARENT metric. Loading computed stats.")
        sys.stdout.flush()
        
        with open(tv_path, encoding="utf8", mode="r") as f:
            TABLE_VALUES = [set(l) for l in json.load(f)]

        with open(rnc_path, encoding="utf8", mode="r") as f:
            _to_load = json.load(f)
            REF_NGRAM_COUNTS = [
                {int(order): {tuple(ngram.split()): float(count) for ngram, count in counter.items()}
                 for order, counter in order2counter.items()}
                for order2counter in _to_load
            ]

        with open(rnw_path, encoding="utf8", mode="r") as f:
            _to_load = json.load(f)
            REF_NGRAM_WEIGHTS = [
                {int(order): {tuple(ngram.split()): float(weight) for ngram, weight in weighter.items()}
                 for order, weighter in order2weighter.items()}
                for order2weighter in _to_load
            ]
            
        self.metric = PARENT(TABLE_VALUES, REF_NGRAM_COUNTS, REF_NGRAM_WEIGHTS)
        
        with open(ref_path, encoding="utf8", mode="r") as f:
            self.references = [ref.strip() for ref in f if ref.strip()]
                
        logger.info("PARENT metric Initialized.")
        sys.stdout.flush()
    
    def __call__(self, batch, rl_forward, baseline_forward):
        """
        There's no better way for now than a for-loop...
        """
        
        rl_sentences, rl_log_probs, rl_attns = rl_forward
        baseline_sentences, baseline_log_probs, baseline_attns = baseline_forward
        
        device = batch.tgt.device
        
        rl_lengths = list()
        baseline_lengths = list()
        decoded_sequences = list()
        rl_scores = list()
        baseline_scores = list()
        for b in range(batch.batch_size):
            
            rl_candidate, rl_length = self.cleaner.clean_candidate_tokens(rl_sentences[:, b],
                                                                    batch.src_map[:, b], 
                                                                    batch.src_ex_vocab[b],
                                                                    rl_attns[:, b])
            baseline_candidate, baseline_length = self.cleaner.clean_candidate_tokens(baseline_sentences[:, b],
                                                                    batch.src_map[:, b], 
                                                                    batch.src_ex_vocab[b],
                                                                    baseline_attns[:, b])
            
            if rl_length == 0:
                rl_length = 1
            rl_lengths.append(rl_length)
            baseline_lengths.append(baseline_length)
            decoded_sequences.append((self.references[batch.indices[b].item()],
                                      " ".join(baseline_candidate), " ".join(rl_candidate)))

            rl_scores.append(self.metric(rl_candidate, batch.indices[b].item()))
            baseline_scores.append(self.metric(baseline_candidate, batch.indices[b].item()))
            
        rl_lengths = torch.LongTensor(rl_lengths).to(device)
        baseline_lengths = torch.LongTensor(baseline_lengths).to(device)
        mask = sequence_mask(rl_lengths, max_len=len(rl_sentences))
        
        sequences_scores = rl_log_probs.masked_fill(~mask.transpose(0,1), 0)
        sequences_scores = sequences_scores.sum(dim=0) / rl_lengths.float()
        
        # we reward the model according to f1_score
        
        rl_rewards = torch.FloatTensor(rl_scores).to(device)
        baseline_rewards = torch.FloatTensor(baseline_scores).to(device)
        rewards = baseline_rewards - rl_rewards
    
        loss = (rewards * sequences_scores).mean()
        stats = self._stats(loss, baseline_rewards.mean(), rl_rewards.mean(),
                            baseline_lengths, rl_lengths,
                            decoded_sequences)
        
        return loss, stats
    
    def _stats(self, loss, baseline_rewards, rl_rewards, 
               baseline_lengths, rl_lengths, decoded_sequences):
        
        loss = loss.item()
        lengths = baseline_lengths.sum().item() + rl_lengths.sum().item()
        
        return onmt.rl_trainer.RLStatistics(rl_loss=loss, rl_rewards=rl_rewards.item(), 
                                            baseline_rewards=baseline_rewards.item(),
                                            n_tgt_words=lengths, 
                                            decoded_sequences=decoded_sequences)