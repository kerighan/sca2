"""Token classes for a LOSS BREAKDOWN BY WHAT THE TOKEN IS, not where it sits.

Motivation (CATCHUP.md, conjecture 1). The aggregate val loss and the per-position
profile both say WHEN generation 3's lead over GDN closes; neither says on WHICH
tokens. The hash-vs-metric-keys conjecture makes a token-level prediction:

    a phase-coded (torus) memory retrieves an EXACT repeat well, so the loss on a
    word that already occurred in the window should stay ahead, while the loss on
    a word seen for the FIRST time in the window -- where only similarity to other
    contexts can help -- is where a dot-product (sphere) memory should catch up.

So the split that matters is not identifier/keyword/punctuation alone but
    word_rep   word-like target that already appears earlier in the window
    word_new   word-like target that does not
and the rest is kept because it costs nothing and separates "local syntax"
(punct, ws, kw) from "content" (word_*, num).

STATIC classes come from the decoded BPE piece alone (one table per vocab, built
once); the rep/new split needs the window, see `split_repeat`. String and comment
contents are NOT recognised (that would need a real tokenizer pass); their pieces
fall into word_*/punct by shape, which is acceptable for a relative comparison
between two models on the same stream.
"""
import keyword
import re
import torch

NAMES = ["word_new", "word_rep", "kw", "punct", "num", "ws", "other"]
WORD_NEW, WORD_REP, KW, PUNCT, NUM, WS, OTHER = range(len(NAMES))
_WORD = 0    # placeholder static class; split into new/rep per window

_KW = set(keyword.kwlist) | {
    "self", "cls", "None", "True", "False", "print", "len", "range", "str", "int",
    "float", "list", "dict", "set", "tuple", "isinstance", "super", "object",
    "type", "Exception", "ValueError", "TypeError", "KeyError", "__init__",
}
_re_word = re.compile(r"^\s?[A-Za-z_][A-Za-z0-9_]*$")
_re_num = re.compile(r"^\s?[0-9][0-9_.eExXa-fA-F]*$")
_re_ws = re.compile(r"^\s+$")
_re_punct = re.compile(r"^\s?[^\sA-Za-z0-9_]+$")


def static_table(tok) -> torch.Tensor:
    """vocab-id -> static class (WORD_NEW used as 'word', split later)."""
    V = tok.get_vocab_size()
    tab = torch.full((V,), OTHER, dtype=torch.long)
    for i in range(V):
        s = tok.decode([i])
        if not s:
            continue
        if _re_ws.match(s):
            tab[i] = WS
        elif _re_word.match(s):
            tab[i] = KW if s.strip() in _KW else WORD_NEW
        elif _re_num.match(s):
            tab[i] = NUM
        elif _re_punct.match(s):
            tab[i] = PUNCT
    return tab


def load_table(prefix="pycode_bpe16k"):
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer(prefix + "-vocab.json", prefix + "-merges.txt")
    return static_table(tok)


def split_repeat(x: torch.Tensor, y: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
    """Per-target class (B,T). x is the input window, y = x shifted by one.

    A word target y[b,t] is WORD_REP iff the SAME token id occurs in x[b,:t+1],
    i.e. among the tokens the model has seen when it predicts it. (T,T) compare
    per row; at B=8, T=1024 that is 8M booleans, negligible next to a forward.
    """
    c = tab.to(y.device)[y]
    B, T = y.shape
    seen = (x[:, None, :] == y[:, :, None])                       # (B,t,s): y_t == x_s
    seen = seen & torch.ones(T, T, dtype=torch.bool, device=y.device).tril()[None]  # s <= t
    rep = seen.any(-1)
    return torch.where((c == WORD_NEW) & rep, torch.full_like(c, WORD_REP), c)


def class_means(loss: torch.Tensor, cls: torch.Tensor):
    """Sum of loss and count per class, for accumulation across batches."""
    K = len(NAMES)
    s = torch.zeros(K, device=loss.device).index_add_(0, cls.flatten(), loss.flatten())
    n = torch.zeros(K, device=loss.device).index_add_(0, cls.flatten(),
                                                      torch.ones_like(loss.flatten()))
    return s, n
