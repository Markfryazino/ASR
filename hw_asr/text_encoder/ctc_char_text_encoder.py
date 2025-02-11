from typing import List, NamedTuple
from collections import defaultdict
from pyctcdecode import build_ctcdecoder

import torch
import numpy as np

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm_path=None, alpha=0.5, beta=1e-4):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.pyctc = build_ctcdecoder(
            labels=[''] + list("".join(self.alphabet).upper()),
            kenlm_model_path=lm_path,
            alpha=alpha,
            beta=beta, 
        )

    def ctc_decode(self, inds: List[int]) -> str:
        last_token = 0
        result = ""
        for ind in inds:
            if ind == last_token:
                continue

            if ind != 0:
                result += self.ind2char[ind]
            
            last_token = ind
        
        return result

    def _extend_and_merge(self, next_char_probs, src_paths):
        new_paths = defaultdict(float)
        for next_char_ind, next_char_prob in enumerate(next_char_probs):
            next_char = self.ind2char[next_char_ind]

            for (text, last_char), path_prob in src_paths.items():
                new_prefix = text if next_char == last_char else text + next_char
                new_prefix = new_prefix.replace(CTCCharTextEncoder.EMPTY_TOK, '')
                new_paths[(new_prefix, next_char)] += path_prob * next_char_prob
        return new_paths

    def _truncate_beam(self, paths, beam_size):
        return dict(sorted(paths.items(), key=lambda x: x[1])[-beam_size:])

    def custom_ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        probs = probs[:probs_length]

        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        paths = {('', CTCCharTextEncoder.EMPTY_TOK): 1.0}
        for next_char_probs in probs:
            paths = self._extend_and_merge(next_char_probs, paths)
            paths = self._truncate_beam(paths, beam_size)
        
        for path, prob in paths.items():
            hypos.append(Hypothesis(path[0], prob))

        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        probs = probs[:probs_length].cpu().detach().numpy()
        decoded_beams = self.pyctc.decode_beams(probs, beam_width=beam_size)

        for text, lm_state, words, logits, lm_logits in decoded_beams:
            hypos.append(Hypothesis(text.lower(), np.exp(lm_logits)))

        return hypos
