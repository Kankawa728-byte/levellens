"""Two-layer BiLSTM with GloVe 300d word vectors and attention pooling.

    python -m levellens.run data=math model=bilstm head=corn

    words -> GloVe vectors -> BiLSTM (2 layers, both directions) -> attention pooling -> head

- The vocabulary is the words of the training set. A word that GloVe knows starts from its GloVe
  vector; a word that GloVe does not know but that occurs at least `min_count` times starts from a
  small random vector; every other word becomes `<unk>`. The vectors are trained further.
- Attention pooling: one learned score per word, softmax over the words of the question, and the
  question vector is the weighted mean of the BiLSTM states. So the model chooses which words count,
  instead of using only the last state.
- The training loop (class weights, early stopping on dev QWK, saving) is in `levellens/models/neural.py`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from levellens.features.glove import DIM, glove_vectors, tokenize
from levellens.models.base import texts
from levellens.models.neural import NeuralModel

PAD, UNK = "<pad>", "<unk>"
VOCAB_FILE = "vocab.txt"


class BiLSTMEncoder(nn.Module):
    """Embedding, BiLSTM and attention pooling: token ids [n, T] -> question vectors [n, 2 * hidden]."""

    def __init__(self, vocab_size: int, hidden: int, layers: int, dropout: float, embedding_dropout: float):
        """Build the embedding, the LSTM and the attention scorer."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, DIM, padding_idx=0)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.lstm = nn.LSTM(DIM, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attention = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.out_features = 2 * hidden

    def attention_weights(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the weight of each word, shape [n, T]. Padding gets weight 0; each row sums to 1."""
        scores = self.attention(states).squeeze(-1).masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return one vector per question: the attention-weighted mean of the BiLSTM states."""
        embedded = self.embedding_dropout(self.embedding(input_ids))
        # packing tells the LSTM where each question ends, so it never reads padding
        packed = pack_padded_sequence(embedded, mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False)
        states, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=input_ids.shape[1])
        weights = self.attention_weights(states, mask)
        return (weights.unsqueeze(-1) * states).sum(dim=1)


class BiLSTM(NeuralModel):
    """The recurrent model of the comparison. Settings are in `configs/model/bilstm.yaml`."""

    DEFAULTS = {
        "hidden": 256,              # LSTM units per direction
        "layers": 2,
        "embeddings": "glove",      # glove, or random (no download; for tests)
        "embedding_dropout": 0.3,
        "lstm_dropout": 0.3,        # between the two LSTM layers
        "max_tokens": 400,          # longer questions are cut
        "min_count": 2,             # a word unknown to GloVe needs this many occurrences to get its own vector
        "lr": 1e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "grad_clip": 5.0,
        "dropout": 0.3,
        "batch_size": 64,
        "eval_batch_size": 256,
        "epochs": 20,
        "patience": 4,
    }

    def prepare(self, train, config) -> None:
        """Build the vocabulary from the training words and load their GloVe vectors."""
        counts = Counter(word for text in texts(train) for word in tokenize(text))
        vectors = glove_vectors(counts) if self.params["embeddings"] == "glove" else {}
        kept = sorted(word for word, count in counts.items() if word in vectors or count >= self.params["min_count"])
        self.vocab = [PAD, UNK, *kept]
        self.index = {word: row for row, word in enumerate(self.vocab)}
        self._glove = vectors
        self.glove_coverage = sum(counts[w] for w in vectors) / max(1, sum(counts.values()))
        print(f"  vocabulary {len(self.vocab):,} words; GloVe covers {self.glove_coverage:.1%} of the training tokens", flush=True)

    def encode(self, items) -> list[list[int]]:
        """Return the word indices of each item, cut to `max_tokens`. An unknown word becomes `<unk>`."""
        unk, limit = self.index[UNK], self.params["max_tokens"]
        return [[self.index.get(word, unk) for word in tokenize(text)[:limit]] or [unk] for text in texts(items)]

    def build_encoder(self) -> BiLSTMEncoder:
        """Build the encoder and copy the GloVe vectors into its embedding layer."""
        p = self.params
        encoder = BiLSTMEncoder(len(self.vocab), p["hidden"], p["layers"], p["lstm_dropout"], p["embedding_dropout"])
        with torch.no_grad():
            encoder.embedding.weight.normal_(0.0, 0.1)
            encoder.embedding.weight[0].zero_()
            for word, vector in getattr(self, "_glove", {}).items():
                encoder.embedding.weight[self.index[word]] = torch.from_numpy(vector)
        self._glove = {}  # the vectors now live in the embedding layer
        return encoder

    def extra_state(self, folder: Path) -> dict:
        """Write the vocabulary into `folder`."""
        (folder / VOCAB_FILE).write_text("\n".join(self.vocab), encoding="utf-8")
        return {"vocab_size": len(self.vocab)}

    def restore_extra(self, folder: Path, state: dict) -> None:
        """Read the vocabulary from `folder`."""
        self.vocab = (folder / VOCAB_FILE).read_text(encoding="utf-8").split("\n")
        self.index = {word: row for row, word in enumerate(self.vocab)}
