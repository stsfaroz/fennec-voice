# From https://huggingface.co/ozonetg/bargein-classifier-ettin-150m (MIT License, see NOTICE). Unchanged.
"""The canonical text normaliser for the barge-in classifier. Train = inference.

Calibrated to the measured output of a CTC telephony ASR (Parakeet-TDT-CTC-110M family). Its raw output is already
lowercase a-z + space with numbers as spoken words, so normalise() is (almost) the identity on it; the rules below
make WRITTEN forms come out the way the ASR prints the same speech: phone groups digit by digit, `dollars and N cents`,
`a.m.` -> `am`, `ok` -> `okay`, `alright` -> `all right`, `lemme` -> `let me`, `wanna` -> `want to` (`gonna`,
`gotta`, `kinda` are kept: the ASR keeps them). The ASR's optional `and` inside numbers (`two thousand and
nineteen`, measured 5/8 after `thousand`) is NEVER stripped here, because runtime text is raw ASR output and a
strip would have to run at inference too; normalise() renders digits without it (except money, 4/4 measured).

format_input.py imports it for every model input. Never copy these functions; a second normaliser is a
train/inference skew.

    normalise(text)        -> normalised caller text (lowercase, apostrophe mapping don't/don t/dont -> don t,
                              accent folding, no punctuation, numbers as words, fillers kept, whitespace collapsed).
                              Idempotent.
    normalise_agent(text)  -> the "light" agent-text form (same word mapping, but , . ? ! ; : kept as tokens).
    int_to_words(n), numbers_to_words(text)
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------------------------------------------
# Numbers as words
# ---------------------------------------------------------------------------------------------------------------
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth"}


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + int_to_words(n % 100))
    for scale, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= scale:
            rest = n % scale
            return int_to_words(n // scale) + " " + name + ("" if rest == 0 else " " + int_to_words(rest))
    raise AssertionError(n)


def _ordinal(n: int) -> str:
    w = int_to_words(n).split()
    last = w[-1]
    if last in _ORD:
        w[-1] = _ORD[last]
    elif last.endswith("y"):
        w[-1] = last[:-1] + "ieth"
    else:
        w[-1] = last + "th"
    return " ".join(w)


def _digits(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def _year(n: int) -> str:
    if 2000 <= n <= 2009:
        return int_to_words(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return int_to_words(hi) + " hundred"
    return int_to_words(hi) + (" oh " + _ONES[lo] if lo < 10 else " " + int_to_words(lo))


def _number(s: str) -> str:
    """A bare digit run -> words: long or zero-led runs digit by digit, 4-digit years year-style, else cardinal."""
    if len(s) >= 5 or (len(s) > 1 and s[0] == "0"):
        return _digits(s)
    n = int(s)
    if len(s) == 4 and (1100 <= n <= 1999 or 2000 <= n <= 2099):
        return _year(n)
    return int_to_words(n)


# Phone-like digit groups (measured: `212-555-0187` -> `two one two five five five zero one eight seven`, 4/4):
# a (3-digit) area code in parentheses, or 3+ groups joined by - or ., or two groups of 7+ digits joined by -
# or . (`555-0187`), are read digit by digit. `5-10` (a range) and `3.5` (a decimal) are not phone groups.
_PHONE = re.compile(r"(?<![\d$])(?:\+?1[-. ]?)?\(\d{3}\)\s?\d{3}[-. ]?\d{4}(?!\d)"
                    r"|(?<![\d$.,])\d+(?:[-.]\d+){2,}(?![\d])"
                    r"|(?<![\d$.,])\d{3}[-.]\d{4}(?![\d])")
_AMPM = re.compile(r"(?<![a-z])([ap])\.\s?m\.?(?![a-z])")        # 9 a.m. / 9 p. m. -> 9 am (measured: `nine am`)


def numbers_to_words(t: str) -> str:
    t = _PHONE.sub(lambda m: " " + _digits(re.sub(r"\D", "", m.group(0))) + " ", t)

    def cardinal_or_number(d: str) -> str:
        return int_to_words(int(d.replace(",", ""))) if "," in d else _number(d)

    # money: $12 / $12.50 / $0.99 / $1,200.50
    def money(m):
        d, c = m.group(1), m.group(2)
        out = []
        if int(d.replace(",", "")) != 0 or not c:
            out.append(cardinal_or_number(d) + (" dollar" if d == "1" else " dollars"))
        if c and int(c) != 0:
            cents = int_to_words(int(c)) + (" cent" if int(c) == 1 else " cents")
            out.append(("and " if out else "") + cents)     # measured: "forty five dollars and fifty cents" (4/4)
        return " " + " ".join(out) + " "
    t = re.sub(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?(?!\d|,\d)", money, t)
    # grouped thousands are cardinals, never years: 1,500 -> one thousand five hundred
    t = re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?!\d|,\d)", lambda m: " " + cardinal_or_number(m.group(0)) + " ", t)
    # times: 3:30 -> three thirty, 9:05 -> nine oh five, 2:00 -> two
    def clock(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return " " + int_to_words(h) + " "
        return " " + int_to_words(h) + (" oh " + _ONES[mi] if mi < 10 else " " + int_to_words(mi)) + " "
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", clock, t)
    t = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: " " + _ordinal(int(m.group(1))) + " ", t, flags=re.I)
    t = re.sub(r"(\d+)\.(\d+)", lambda m: " " + _number(m.group(1)) + " point " + _digits(m.group(2)) + " ", t)
    t = re.sub(r"(\d)\s?%", r"\1 percent", t)
    t = re.sub(r"\d+", lambda m: " " + _number(m.group(0)) + " ", t)                  # also splits k7 -> k seven
    return t


# ---------------------------------------------------------------------------------------------------------------
# Apostrophes: don't / don t / dont -> "don t". Only unambiguous apostrophe-less forms are mapped
# (never its, well, were, ill, id, lets, wed, hell, shell, cant is mapped: "cant" as a noun is negligible on calls).
# ---------------------------------------------------------------------------------------------------------------
_BARE_CONTRACTIONS = {
    "dont": "don t", "cant": "can t", "wont": "won t", "didnt": "didn t", "doesnt": "doesn t", "isnt": "isn t",
    "arent": "aren t", "wasnt": "wasn t", "werent": "weren t", "couldnt": "couldn t", "wouldnt": "wouldn t",
    "shouldnt": "shouldn t", "havent": "haven t", "hasnt": "hasn t", "hadnt": "hadn t", "aint": "ain t",
    "mustnt": "mustn t", "neednt": "needn t", "im": "i m", "ive": "i ve", "youre": "you re", "youve": "you ve",
    "youll": "you ll", "youd": "you d", "theyre": "they re", "theyve": "they ve", "theyll": "they ll",
    "theyd": "they d", "weve": "we ve", "thats": "that s", "whats": "what s", "wheres": "where s", "whos": "who s",
    "heres": "here s", "theres": "there s", "hows": "how s", "whered": "where d", "whatd": "what d",
    "itll": "it ll", "thatll": "that ll", "shes": "she s", "yall": "y all", "howd": "how d", "whyd": "why d",
    "couldve": "could ve", "wouldve": "would ve", "shouldve": "should ve", "mightve": "might ve",
}
# Written forms the ASR never prints, mapped to what it prints for the same speech (ASR round trip, 4/4 unless
# noted): `OK` -> `okay`, `Alright` -> `all right`, `lemme` -> `let me`, `wanna` -> `want to` (2/4; the other 2/4
# were the garble `wantna`, which is kept as a token). `gonna`, `gotta`, `kinda` are printed as such and kept.
# `gimme`, `gotcha`, `aight` were not measured and are left alone.
_ASR_FORMS = {"ok": "okay", "alright": "all right", "lemme": "let me", "wanna": "want to"}
_WORD_MAP = {**_BARE_CONTRACTIONS, **_ASR_FORMS}
_APOSTROPHES = "'’‘ʼ`´"


def _fold(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    for a in _APOSTROPHES:
        t = t.replace(a, "'")
    return t


def normalise(text: Optional[str]) -> str:
    """normalise(): lowercase; `don't`, `don t`, `dont` -> `don t`; strip the remaining punctuation;
    numbers as words; collapse whitespace; KEEP fillers. Idempotent."""
    if not text:
        return ""
    t = _pre(text)
    t = re.sub(r"[^a-z ]+", " ", t)
    words = [_WORD_MAP.get(w, w) for w in t.split()]
    return " ".join(words)


def _pre(text: str) -> str:
    """Shared first steps of normalise()/normalise_agent(): fold, lowercase, `&`, `a.m.`/`o.k.`, numbers, apostrophe."""
    t = _fold(text).lower()
    t = t.replace("&", " and ")
    t = _AMPM.sub(r"\1m", t)
    t = re.sub(r"(?<![a-z])o\.\s?k\.?(?![a-z])", "okay", t)
    t = numbers_to_words(t)
    return t.replace("'", " ")


def normalise_agent(text: Optional[str]) -> str:
    """Agent-text form: the normalise() word mapping, but `, . ? ! ; :` survive as separate tokens (the question
    mark and clause breaks are cues for the model; agent text is TTS input, not ASR output)."""
    if not text:
        return ""
    t = _pre(text)
    t = re.sub(r"([,.?!;:])", r" \1 ", t)
    t = re.sub(r"[^a-z,.?!;: ]+", " ", t)
    words = [_WORD_MAP.get(w, w) for w in t.split()]
    out = []
    for w in words:                      # collapse runs of punctuation (e.g. "?." or ". ." from ellipses)
        if w in ",.?!;:" and out and out[-1] in ",.?!;:":
            if w == "?":
                out[-1] = "?"
            continue
        out.append(w)
    return " ".join(out)
