"""Seed/connector/name role classification for the NER pipeline."""

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache

log = logging.getLogger(__name__)

MIN_COUNT      = 3
SEED_THRESHOLD = 0.5


@dataclass
class RoleParams:
    """Parameters controlling seed/connector/name classification.

    role_mode:
      "frequency"  — seed/connector split by Brown-corpus frequency + entity rate (original)
      "presence"   — seed/connector split by how many datasets a word appears as entity in
                     (seed = 1 dataset only, connector = 2+)
      "variance"   — seed/connector by mean_entity_rate - λ * std_entity_rate across datasets
    """
    brown_freq: dict
    max_brown_freq: float
    lambda_seed: float = 0.3
    lambda_oov: float = 0.1
    caps_threshold: float = 0.5
    role_mode: str = "frequency"           # "frequency" | "presence" | "variance"
    dataset_presence: dict = field(default_factory=dict)  # word.lower() -> n_datasets
    cross_dataset_rates: dict = field(default_factory=dict)  # word.lower() -> (mean_rate, std_rate)

    def brown_norm(self, word: str, force_lower: bool = False) -> float:
        if force_lower:
            lookup = word.lower()
        else:
            lookup = word if word.isupper() else word.lower()
        freq = self.brown_freq.get(lookup, 0.0)
        return math.log(freq + 1) / math.log(self.max_brown_freq + 1)

    def score(self, entity_rate: float, word: str, force_lower: bool = False) -> float:
        return entity_rate - self.lambda_seed * self.brown_norm(word, force_lower)


@lru_cache(maxsize=None)
def build_brown_freq() -> tuple[dict, float]:
    """Load NLTK Brown corpus word frequencies (cached after first call)."""
    from collections import Counter
    from nltk.corpus import brown
    counts = Counter(w.lower() for w in brown.words())
    total  = sum(counts.values())
    freq   = {w: c / total * 1_000_000 for w, c in counts.items()}
    return freq, max(freq.values())



def _is_mostly_caps(tokens: list[str], threshold: float = 0.5) -> bool:
    """True if ≥ threshold of all alphabetical characters in the sentence are uppercase."""
    letters = [c for t in tokens for c in t if c.isalpha()]
    if not letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= threshold


def _force_lower(tokens: list[str], idx: int, mostly_caps: bool) -> bool:
    """True if Brown lookup should use word.lower() for this token.

    Covers two cases beyond the default isupper() guard:
    - the whole sentence is mostly caps
    - the word is title-cased at a grammatical sentence boundary (position 0
      or immediately after a sentence-ending punctuation mark), where the
      capitalisation is structural rather than semantic.
    """
    if mostly_caps:
        return True
    word = tokens[idx]
    if word and word[0].isupper() and not word.isupper():
        if idx == 0:
            return True
        if tokens[idx - 1][-1:] in ".!?":
            return True
    return False


def _role(binary: int, count: int, rate: float, rp: RoleParams, word: str,
          force_lower: bool = False) -> str:
    """Classify a token instance into non_entity / seed / connector.

    frequency : seed/connector by Brown frequency + entity rate (original logic)
    presence  : seed   = word is an entity in exactly 1 dataset (hard to transfer)
                connector = word is an entity in 2+ datasets (easy to transfer)
    variance  : seed_score = mean_entity_rate - λ * std_entity_rate across datasets
                seed if seed_score >= SEED_THRESHOLD; OOV falls back to brown_norm
    """
    if binary == 0:
        return "non_entity"
    if rp.role_mode == "presence":
        n = rp.dataset_presence.get(word.lower(), 0)
        return "seed" if n <= 1 else "connector"
    if rp.role_mode == "variance":
        entry = rp.cross_dataset_rates.get(word.lower())
        if entry is None:
            # OOV: not seen with enough count in any dataset — use brown_norm proxy
            bn = rp.brown_norm(word, force_lower)
            return "seed" if bn <= rp.lambda_oov else "connector"
        mean_r, std_r = entry
        seed_score = mean_r - rp.lambda_seed * std_r
        return "seed" if seed_score >= SEED_THRESHOLD else "connector"
    # frequency mode
    bn = rp.brown_norm(word, force_lower)
    if count < MIN_COUNT:
        return "seed" if bn <= rp.lambda_oov else "connector"
    score = rate - rp.lambda_seed * bn
    return "seed" if score >= SEED_THRESHOLD else "connector"



def role_fingerprint(rp: RoleParams) -> str:
    """Short string uniquely identifying role params — used as cache sub-directory."""
    return f"{rp.role_mode}_s{rp.lambda_seed:.3f}_o{rp.lambda_oov:.3f}"
