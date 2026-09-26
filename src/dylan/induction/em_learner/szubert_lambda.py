"""Normalize Szubert / UDepLambda CHILDES formulae into the Eve lambda dialect.

The published Adam file (``Lou1sM/CHILDES_UD2LF_2``, ``adam.all_lf.txt``)
is the output of their UD-to-lambda converter. :func:`convert_lambda`
already rewrites the Eve dialect to TTR, so this module only renames
binders, part-of-speech tags, and a few wrappers (``BARE``, ``att``)
before that call.

"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dylan.induction.em_learner.lambda_ttr_converter import convert_lambda

# Longer tags first so ``pro:per`` is not left behind a shorter prefix.
_POS_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("det:art|", "det|"),
    ("det:num|", "qn|"),
    ("det:dem|", "det|"),
    ("det:poss|", "pro:poss:det|"),
    ("pro:per|", "pro|"),
    ("pro:sub|", "pro|"),
    ("pro:obj|", "pro|"),
    ("pro:int|", "pro|"),
    ("pro:rel|", "pro|"),
    ("pro:indef|", "pro|"),
    ("pro:exist|", "pro|"),
    ("pro:refl|", "pro|"),
    ("pro:poss|", "pro|"),
    ("mod:aux|", "aux|"),
    ("mod|", "aux|"),
    ("cop|", "aux|"),
    ("post|", "adv|"),
    ("v:obj|", "v|"),
    ("coord|", "conj|"),
    ("n:gerund|", "n|"),
    ("n:let|", "n|"),
    ("n:pt|", "n|"),
    ("chi|", "n:prop|"),
    ("poss|", "pro:poss:det|"),
    ("co|", "adv|"),
)

_BINDER = re.compile(r"lambda\s+\$([0-9]+)_\{[^}]+\}")
_PREDICATE_NAME = re.compile(r"\|([A-Za-z0-9~.+&_-]+)(\()")
_EMBEDDED_LAMBDA = re.compile(r"lambda\s+\$([0-9]+)_\{[^}]+\}\.")


def _call_span(text: str, name: str, start: int = 0) -> tuple[int, int, list[str]] | None:
    """Find ``name(...)`` and return its span plus top-level arguments.

    :param text: Formula fragment.
    :param name: Call name, without the parenthesis.
    :param start: Index to search from.
    :returns: ``(begin, end, args)`` or ``None`` when no call remains.
    """
    needle = name + "("
    at = text.find(needle, start)
    if at < 0:
        return None
    index = at + len(needle)
    depth = 1
    arg_start = index
    args: list[str] = []
    while index < len(text) and depth:
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                args.append(text[arg_start:index])
                return at, index + 1, args
        elif character == "," and depth == 1:
            args.append(text[arg_start:index])
            arg_start = index + 1
        index += 1
    return None


def _replace_calls(
    text: str,
    name: str,
    rewrite: Callable[[list[str]], str | None],
) -> str:
    """Replace every balanced call ``name(...)`` via *rewrite*.

    :param text: Formula.
    :param name: Call name.
    :param rewrite: Maps the argument list to a replacement string, or ``None`` to keep it.
    :returns: The rewritten formula.
    """
    cursor = 0
    pieces: list[str] = []
    while True:
        found = _call_span(text, name, cursor)
        if found is None:
            pieces.append(text[cursor:])
            break
        begin, end, args = found
        pieces.append(text[cursor:begin])
        replacement = rewrite(args)
        if replacement is None:
            inner = ",".join(_replace_calls(arg, name, rewrite) for arg in args)
            pieces.append(f"{name}({inner})")
            cursor = end
        else:
            pieces.append(_replace_calls(replacement, name, rewrite))
            cursor = end
    return "".join(pieces)


def _strip_indices(text: str) -> str:
    """Drop word-position suffixes such as ``eat_4`` and ``leonard_4_lion_5``.

    :param text: Formula that still has Szubert token indices.
    :returns: The same formula with every ``_`` plus digits removed.
    """
    return re.sub(r"_\d+", "", text)


def _bare_you(text: str) -> str:
    """Tag a bare ``you`` as ``pro|you``.

    Szubert leaves the addressee untagged. The Eve rules only accept ``pro|you``.

    :param text: Formula after part-of-speech renaming.
    :returns: The formula with bare ``you`` tokens tagged.
    """
    return re.sub(r"(?<![A-Za-z0-9_|:~])you(?![A-Za-z0-9_])", "pro|you", text)


def _event_variable(text: str) -> str:
    """Return the matrix event binder, or ``$0`` when the formula does not mark one.

    The matrix event is the first ``_{ev}`` binder outside any parenthesis.
    An embedded complement uses a later binder and is not the matrix event.

    :param text: Formula whose binders already use ``_{ev}``.
    :returns: A variable such as ``$0`` or ``$1``.
    """
    for match in re.finditer(r"lambda\s+\$([0-9]+)_\{ev\}", text):
        depth = text[: match.start()].count("(") - text[: match.start()].count(")")
        if depth == 0:
            return "$" + match.group(1)
    found = re.findall(r"lambda\s+\$([0-9]+)_\{ev\}", text)
    if found:
        return "$" + found[0]
    return "$0"


def _close_question_negation(text: str) -> str:
    """Give ``Q`` and ``not`` the outer event argument Eve formulae use.

    Szubert writes ``Q(v|go(pro|you,$0))``. The converter wants
    ``Q(v|go(pro|you,$0),$0)``, with the event kept inside the verb as well.

    :param text: Formula after wrapper unwrapping.
    :returns: The formula with a copied event argument on one-place ``Q`` / ``not``.
    """
    event = _event_variable(text)

    def _rewrite(name: str) -> Callable[[list[str]], str | None]:
        def _one(args: list[str]) -> str | None:
            if len(args) != 1:
                return None
            arg = args[0].strip()
            if not arg or re.fullmatch(r"\$[0-9]+", arg):
                return None
            return f"{name}({arg},{event})"

        return _one

    text = _replace_calls(text, "Q", _rewrite("Q"))
    return _replace_calls(text, "not", _rewrite("not"))


def _discourse_no(text: str) -> str:
    """Retag ``qn|no($event)`` as the adverb ``no``.

    :param text: Formula after quantifier renaming.
    :returns: The formula with a discourse ``no`` marked ``adv``.
    """

    def _rewrite(args: list[str]) -> str | None:
        if len(args) == 1 and re.fullmatch(r"\$[0-9]+", args[0].strip()):
            return f"adv|no({args[0].strip()})"
        return None

    return _replace_calls(text, "qn|no", _rewrite)


def _pronoun_one(text: str) -> str:
    """Retag ``pro|one($n)`` as the noun ``one``.

    :param text: Formula after pronoun renaming.
    :returns: The formula with a restricted ``one`` marked ``n``.
    """

    def _rewrite(args: list[str]) -> str | None:
        if len(args) == 1 and re.fullmatch(r"\$[0-9]+", args[0].strip()):
            return f"n|one({args[0].strip()})"
        return None

    return _replace_calls(text, "pro|one", _rewrite)


def _term_end(text: str, start: int) -> int:
    """Return the index just after the term that begins at *start*.

    :param text: Formula.
    :param start: Index of the first character of a call or atom.
    :returns: End index, or *start* when no term is found.
    """
    if start >= len(text):
        return start
    if text[start] == "(":
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    return index + 1
        return len(text)
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            if depth == 0:
                return index
            depth -= 1
        elif character == "," and depth == 0:
            return index
    return len(text)


def _lower_embedded_lambda(text: str) -> str:
    """Turn an embedded ``lambda`` complement into an Eve ``and`` conjunction.

    ``go(you, lambda $1.sit(you,$1), $0)`` becomes
    ``and(go(you,$0), sit(you,$0))``. The copied event is the formula's
    ``_{ev}`` binder.

    :param text: Formula that may still contain a complement lambda.
    :returns: The formula with those complements lowered.
    """
    event = _event_variable(text)
    guard = 0
    while guard < 20:
        guard += 1
        match = None
        for found in _EMBEDDED_LAMBDA.finditer(text):
            depth = text[: found.start()].count("(") - text[: found.start()].count(")")
            if depth > 0 and "lambda" not in text[found.end() : _term_end(text, found.end())]:
                match = found
                break
        if match is None:
            break
        variable = "$" + match.group(1)
        body_start = match.end()
        body_end = _term_end(text, body_start)
        body = re.sub(
            rf"{re.escape(variable)}(?!\d)",
            event,
            text[body_start:body_end],
        )
        call_start = text.rfind("(", 0, match.start())
        if call_start < 0:
            break
        name_start = call_start
        while name_start > 0 and text[name_start - 1] in (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:|~+&_-"
        ):
            name_start -= 1
        call_end = _term_end(text, name_start)
        if call_end <= call_start:
            break
        inner = text[call_start + 1 : call_end - 1]
        rel_lambda = match.start() - (call_start + 1)
        rel_body_end = body_end - (call_start + 1)
        before = inner[:rel_lambda].strip(" ,")
        after = inner[rel_body_end:].strip(" ,")
        kept = ",".join(part for part in (before, after) if part)
        call = text[name_start:call_start] + "(" + kept + ")"
        replacement = f"and({call},{body})"
        text = text[:name_start] + replacement + text[call_end:]
    return text


def _supply_prep_event(text: str) -> str:
    """Append the event variable to a preposition that has only an object.

    :param text: Formula after complement lowering.
    :returns: The formula with ``prep|of(pro|it)`` rewritten as ``prep|of(pro|it,$0)``.
    """
    event = _event_variable(text)
    cursor = 0
    pieces: list[str] = []
    while True:
        at = text.find("prep|", cursor)
        if at < 0:
            pieces.append(text[cursor:])
            break
        paren = text.find("(", at)
        if paren < 0 or "|" in text[at + 5 : paren]:
            pieces.append(text[cursor : at + 5])
            cursor = at + 5
            continue
        name = text[at:paren]
        found = _call_span(text, name, at)
        if found is None or found[0] != at:
            pieces.append(text[cursor : at + 5])
            cursor = at + 5
            continue
        _begin, end, args = found
        pieces.append(text[cursor:at])
        if args and not re.fullmatch(r"\$[0-9]+", args[-1].strip()):
            pieces.append(name + "(" + ",".join(args) + f",{event})")
        else:
            pieces.append(text[at:end])
        cursor = end
    return "".join(pieces)


def _map_pos(text: str) -> str:
    """Rename Szubert part-of-speech tags onto the Eve tag set."""
    text = text.replace("pro:exist|here", "adv:loc|here")
    text = text.replace("pro:exist|there", "adv:loc|there")
    for source, target in _POS_REPLACEMENTS:
        text = text.replace(source, target)
    return text


def _map_suffixes(text: str) -> str:
    """Map ``-past`` / ``-presp`` / ``-pl`` onto Eve predicate suffixes."""

    def _one(match: re.Match[str]) -> str:
        name = match.group(1)
        name = name.replace("-presp", "-PROG")
        name = name.replace("-pastp", "&PASTP")
        name = name.replace("-past", "&PAST")
        name = name.replace("-3s", "&3S")
        name = name.replace("-cond", "&COND")
        name = name.replace("-zero", "&ZERO")
        name = name.replace("-pl", "-PL")
        return "|" + name + match.group(2)

    return _PREDICATE_NAME.sub(_one, text)


def _unwrap_bare(text: str) -> str:
    """Replace ``BARE($n, noun)`` with ``noun``."""

    def _rewrite(args: list[str]) -> str | None:
        if len(args) >= 2:
            return args[1]
        return None

    previous = None
    while previous != text:
        previous = text
        text = _replace_calls(text, "BARE", _rewrite)
    return text


def _unwrap_att(text: str) -> str:
    """Replace ``att(head, modifier)`` with ``and(head, modifier)``."""

    def _rewrite(args: list[str]) -> str | None:
        content = [arg for arg in args if not re.fullmatch(r"\$[0-9]+", arg.strip())]
        if len(content) >= 2:
            return "and(" + ",".join(content) + ")"
        if len(content) == 1:
            return content[0]
        return None

    previous = None
    while previous != text:
        previous = text
        text = _replace_calls(text, "att", _rewrite)
    return text


def _unwrap_dollar_applications(text: str) -> str:
    """Unwrap ``$n($m, content)`` and delete ``$n($m)`` adjunct calls.

    :param text: Formula after ``BARE`` and ``att`` unwrapping.
    :returns: The formula with higher-order variable applications removed.
    """
    cursor = 0
    pieces: list[str] = []
    while cursor < len(text):
        match = re.search(r"\$[0-9]+\(", text[cursor:])
        if match is None:
            pieces.append(text[cursor:])
            break
        begin = cursor + match.start()
        pieces.append(text[cursor:begin])
        found = _call_span(text, text[begin : cursor + match.end() - 1], begin)
        # name includes the variable, e.g. "$1"
        name = text[begin : cursor + match.end() - 1]
        found = _call_span(text, name, begin)
        if found is None or found[0] != begin:
            pieces.append(text[begin : begin + 1])
            cursor = begin + 1
            continue
        _start, end, args = found
        if len(args) >= 2 and re.fullmatch(r"\$[0-9]+", args[0].strip()):
            pieces.append(args[1])
            cursor = end
            continue
        if len(args) == 1 and re.fullmatch(r"\$[0-9]+", args[0].strip()):
            cursor = end
            continue
        if (
            len(args) == 2
            and re.fullmatch(r"\$[0-9]+", args[1].strip())
            and not args[0].strip().startswith("$")
        ):
            # ``$1(pro|that,$0)`` is an identity question: eq(wh, that, event).
            pieces.append(f"eq({name},{args[0]},{args[1]})")
            cursor = end
            continue
        pieces.append(text[begin:end])
        cursor = end
    rewritten = "".join(pieces)
    rewritten = re.sub(r",,", ",", rewritten)
    return rewritten


def _single_and(text: str) -> str:
    """Replace ``and(only)`` with ``only``."""

    def _rewrite(args: list[str]) -> str | None:
        content = [arg for arg in args if arg.strip()]
        if len(content) == 1:
            return content[0]
        return None

    previous = None
    while previous != text:
        previous = text
        text = _replace_calls(text, "and", _rewrite)
    return text


def _map_binders(text: str) -> str:
    """Map Szubert binder types onto ``e`` and ``ev``."""
    text = text.replace("_{r}", "_{ev}")
    text = text.replace("_{<r,t>}", "_{e}")
    text = text.replace("_{<<e,e>,e>}", "_{e}")
    return text


def _drop_unused_binders(text: str) -> str:
    """Remove a leading binder whose variable no longer occurs in the body."""
    pieces = text.split(".")
    if len(pieces) < 2:
        return text
    body = pieces[-1]
    kept: list[str] = []
    for binder in pieces[:-1]:
        match = _BINDER.search(binder) or re.search(r"\$([0-9]+)", binder)
        if match is None:
            kept.append(binder)
            continue
        variable = "$" + match.group(1)
        later = ".".join(kept) + "." + body
        if variable not in later:
            continue
        kept.append(binder)
    if not kept:
        return body
    return ".".join(kept + [body])


def normalize_szubert_lambda(semantics: str) -> str:
    """Rewrite one Szubert formula into the Eve dialect ``convert_lambda`` accepts.

    :param semantics: A line such as ``lambda $0_{r}.not(mod|will_2(...))``.
    :returns: The same proposition with Eve binders, tags, and wrappers.
    """
    text = semantics.strip()
    text = _strip_indices(text)
    text = _map_pos(text)
    text = _map_suffixes(text)
    text = _bare_you(text)
    text = _unwrap_bare(text)
    text = _unwrap_att(text)
    text = _unwrap_dollar_applications(text)
    text = _single_and(text)
    text = _map_binders(text)
    text = _lower_embedded_lambda(text)
    text = _discourse_no(text)
    text = _pronoun_one(text)
    text = _supply_prep_event(text)
    text = _close_question_negation(text)
    text = _drop_unused_binders(text)
    text = re.sub(r",,", ",", text)
    text = re.sub(r"\(\s*,", "(", text)
    text = re.sub(r",\s*\)", ")", text)
    return text


def convert_szubert_lambda(semantics: str, utterance: str) -> str:
    """Normalize *semantics* and convert it to a TTR record type.

    :param semantics: Szubert logical form.
    :param utterance: Surface string used for wh-word placement.
    :returns: A TTR record-type string.
    """
    return convert_lambda(normalize_szubert_lambda(semantics), utterance)


@dataclass(frozen=True)
class SzubertPair:
    """One sentence from ``adam.all_lf.txt``."""

    sample_index: int
    utterance: str
    semantics: str

    def source_comment(self, location: str, comparison: bool) -> str:
        """Back-reference plus whether this row is in the paper comparison slice.

        :param location: ``adamN.conll.txt`` and a sample index, or the LF ordinal.
        :param comparison: True when the sentence is in the public learner easy set.
        """
        flag = "comparison=paper" if comparison else "comparison=extra"
        return f"{location} {flag}"


def iter_szubert_pairs(path: str | Path) -> list[SzubertPair]:
    """Read ``Sent`` / ``Sem`` / ``example_end`` blocks.

    :param path: ``adam.all_lf.txt`` or the same layout.
    :returns: Pairs in file order. ``sample_index`` is 1-based in that file.
    """
    pairs: list[SzubertPair] = []
    sent: str | None = None
    sem: str | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("Sent:"):
            sent = line.split(":", 1)[1].strip()
        elif line.startswith("Sem:"):
            sem = line.split(":", 1)[1].strip()
        elif line.startswith("example_end") and sent is not None and sem is not None:
            pairs.append(SzubertPair(len(pairs) + 1, sent, sem))
            sent = sem = None
    return pairs


@dataclass(frozen=True)
class ConllSentence:
    """One dependency tree from a Szubert ``adamN.conll.txt`` split."""

    filename: str
    sample_index: int
    text: str
    has_communicator: bool


def _split_sort_key(path: Path) -> tuple[int, str]:
    """Order ``adam1`` … ``adam41`` numerically; other names last."""
    match = re.search(r"adam(\d+)", path.name)
    if match is None:
        return (10_000, path.name)
    return (int(match.group(1)), path.name)


def iter_conll_sentences(folder: str | Path) -> list[ConllSentence]:
    """Read surface sentences from ``adam*.conll.txt`` in session order.

    Column 3 is the word form. A token whose xpos is ``co`` is a communicator
    (discourse marker). ``sample_index`` is the 1-based tree in that file.

    :param folder: Directory of split CoNLL files.
    """
    sentences: list[ConllSentence] = []
    root = Path(folder)
    paths = sorted(root.glob("adam*.conll.txt"), key=_split_sort_key)
    for path in paths:
        if path.name.startswith("adam.dummy"):
            continue
        sample = 0
        words: list[str] = []
        communicator = False
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        def _flush() -> None:
            """Store one CoNLL tree and reset the token buffer."""
            nonlocal sample, words, communicator
            if not words:
                return
            sample += 1
            sentences.append(
                ConllSentence(path.name, sample, " ".join(words), communicator)
            )
            words = []
            communicator = False

        for line in lines:
            if not line.strip():
                _flush()
                continue
            columns = line.split("\t")
            if len(columns) < 5 or not columns[0].isdigit():
                continue
            words.append(columns[2])
            if columns[4] == "co":
                communicator = True
        _flush()
    return sentences


def align_lf_to_conll(
    pairs: list[SzubertPair],
    sentences: list[ConllSentence],
) -> list[ConllSentence | None]:
    """Match each logical form to the next unused CoNLL sentence with the same text.

    The LF file omits trees the converter rejected, so alignment walks both
    sequences in order and skips CoNLL trees that have no logical form.

    :param pairs: Logical forms in file order.
    :param sentences: CoNLL trees in ``adam1`` … ``adam41`` order.
    :returns: One location per pair, or ``None`` when the string was not found.
    """
    cursor = 0
    located: list[ConllSentence | None] = []
    for pair in pairs:
        target = " ".join(pair.utterance.split())
        found: ConllSentence | None = None
        while cursor < len(sentences):
            current = sentences[cursor]
            cursor += 1
            if " ".join(current.text.split()) == target:
                found = current
                break
        located.append(found)
    return located


def load_comparison_utterances(path: str | Path) -> set[str]:
    """Load surface strings from the acquisition repo's easy Adam file.

    Mahon et al. report 5,320 Adam utterances after dropping communicator
    words. The public learner file ``Adam.all_easy_lf.txt`` is the closest
    released list (5,734 sentences). Membership in that list is the
    comparison flag.

    :param path: Easy LF file with ``Sent:`` lines.
    :returns: Normalized utterance strings.
    """
    found: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("Sent:"):
            found.add(" ".join(line.split(":", 1)[1].split()))
    return found
