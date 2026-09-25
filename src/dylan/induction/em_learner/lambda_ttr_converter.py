"""Lambda-calculus CHILDES formulae to TTR record types.

Faithful port of Java ``qmul.ds.learn.CorpusConverter`` and
``qmul.ds.learn.CorpusConverterAgenda`` (DyLan). The rules target the
Eve ``trainPairs`` logical forms (``lambda $0_{ev}.v|go(pro|you,$0)``),
not raw CHAT transcripts. ``CorpusConverter`` in this package remains the
load/save stub; call :func:`convert_lambda` for the actual rewrite.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from dylan.formula.predicate_argument import PredicateArgumentFormula
from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_record_type import TTRRecordType

logger = logging.getLogger(__name__)

_AGENDA_CAP = 100

_TENSE_PREDICATE = re.compile(r"^([A-P,R-Z]+)(\()(.*\))(\))$")
_CONJUNCT_PREDICATE = re.compile(r"^(and\()(.*)(\))$")
_LAMBDA = re.compile(r"^(lambda[\s])(\$[0-9])(\_\{)(e|ev)(\})$")
_VARIABLE_INIT = re.compile(r"(\()(\$[0-9])(,)")
_FINAL_VARIABLE = re.compile(r"(.*)(,\$[0-9]\))")
_PREDICATE_POS = re.compile(r"^([^|,)\(]*)(\|)([^|,)\(]*)(\()(.*)(\))$")
_PREDICATE_POS_FINAL_VAR = re.compile(
    r"^([^|,()]+)(\|)+([^|,()]+)(\()+(.*,)*(\$[0-9])+(\))+$"
)
_PREDICATE_POS_MULTI = re.compile(
    r"^([^|,])(\|)((\+[^|,]*\|[^|,]*)+)(\()(\$[0-9])(\))$"
)
_PREDICATE_POS_MULTI_PRED = re.compile(
    r"^([^|,])(\|)((\+[^|,]*\|[^|,]*)+)(\()(.*\|.*)(\))$"
)
_ATOMIC_POS = re.compile(r"^([^|,()]*)(\|)([^|,)\(]*)$")
_DUPLICATE_TYPES = re.compile(r"(\+)([^|,+]*)(\|)([^|,+]*)")
_DET_QUANT = re.compile(
    r"^([^|+(]*)(\|)([^|+(]*)(\()(\$[0-9])(,)(.+\(\$[0-9].*)(\))$"
)
_NUMBERED_REST = re.compile(r"(.*)([0-9])(.*)")
_NUMBER_WORDS = (
    "zero_",
    "one_",
    "two_",
    "three_",
    "four_",
    "five_",
    "six_",
    "seven_",
    "eight_",
    "nine_",
)


class LambdaTTRConversionError(Exception):
    """A lambda formula could not be rewritten with the existing rules."""


def _full(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """Return a full-string match, matching Java ``Matcher.matches``."""
    return pattern.fullmatch(text)


def _number_replace(with_numbers: str) -> str:
    """Rewrite digits 0–9 as ``zero_`` … ``nine_`` (Java ``numberReplace``)."""
    for i in range(10):
        if str(i) in with_numbers:
            with_numbers = re.sub(str(i), _NUMBER_WORDS[i], with_numbers)
    return with_numbers


def _normalize_manifest(ttr_string: str) -> str:
    """Lower-case and turn ``-`` / ``&`` into ``_`` before ``TTRField.parse``."""
    return ttr_string.lower().replace("-", "_").replace("&", "_")


def _parse_field(ttr_string: str) -> TTRField | None:
    """Parse one agenda field, or ``None`` when Java would store null."""
    if ttr_string == "":
        parsed = TTRField.parse(ttr_string)
        return parsed
    rewritten = ttr_string
    if "==" in ttr_string:
        label = ttr_string[: ttr_string.index("==")]
        rest = ttr_string[ttr_string.index("==") :]
        ending = ""
        if "(" in rest:
            ending = rest[rest.index("(") :]
            rest = rest[: rest.index("(")]
        if _full(_NUMBERED_REST, rest) and not label.startswith("head") and not label.startswith("r"):
            rewritten = label + _number_replace(rest) + ending
    parsed = TTRField.parse(_normalize_manifest(rewritten))
    if parsed is None and rewritten.strip():
        raise LambdaTTRConversionError(f"TTR field did not parse: {rewritten!r}")
    return parsed


@dataclass
class ConverterAgenda:
    """Graph agenda of partially converted CHILDES nodes (Java ``CorpusConverterAgenda``)."""

    childes_strings: list[str | None] = field(default_factory=lambda: [None] * _AGENDA_CAP)
    ttr_fields: list[TTRField | None] = field(default_factory=lambda: [None] * _AGENDA_CAP)
    resolveds: list[bool] = field(default_factory=lambda: [False] * _AGENDA_CAP)
    parent_node_numbers: list[int] = field(default_factory=lambda: [0] * _AGENDA_CAP)
    new_node: int = -1

    def add_to_agenda(self, childes_string: str, ttr_string: str, parent_node: int) -> int:
        """Append a node and return its index (Java ``addtoAgenda``)."""
        self.new_node += 1
        if self.new_node >= _AGENDA_CAP:
            raise LambdaTTRConversionError(
                f"agenda exceeded Java cap of {_AGENDA_CAP} nodes"
            )
        self.childes_strings[self.new_node] = childes_string
        self.ttr_fields[self.new_node] = _parse_field(ttr_string)
        self.resolveds[self.new_node] = False
        self.parent_node_numbers[self.new_node] = parent_node
        return self.new_node

    def put_ttr_string(self, ttr_string: str | None, node: int) -> None:
        """Overwrite the field at *node* (Java ``putTTRstring``)."""
        if ttr_string is None:
            self.ttr_fields[node] = None
            return
        self.ttr_fields[node] = _parse_field(ttr_string)

    def make_resolved(self, node: int) -> None:
        """Mark *node* resolved (Java ``makeResolved``)."""
        self.resolveds[node] = True

    def remove_node(self, node: int) -> None:
        """Turn *node* into an empty resolved placeholder (Java ``removeNode``)."""
        self.childes_strings[node] = None
        self.ttr_fields[node] = None
        self.resolveds[node] = True
        self.parent_node_numbers[node] = 0

    def resolved(self, node: int) -> bool:
        """Whether *node* is already resolved."""
        return self.resolveds[node]

    def is_complete(self) -> bool:
        """True when every allocated node is resolved (Java ``isComplete``)."""
        if self.new_node == -1:
            return False
        return all(self.resolveds[i] for i in range(self.new_node + 1))

    def get_daughters(self, node: int) -> list[int]:
        """Return child indexes of *node* (Java ``getDaughters``)."""
        return [
            i
            for i in range(node, self.new_node + 1)
            if self.parent_node_numbers[i] == node
        ]

    def all_daughters_resolved(self, node: int) -> bool:
        """True only when *node* has daughters and all of them are resolved."""
        has_daughters = False
        for i in range(node, self.new_node + 1):
            if self.parent_node_numbers[i] == node:
                has_daughters = True
                if not self.resolveds[i]:
                    return False
        return has_daughters

    def resolve_restrictor(self, node: int) -> None:
        """Fold resolved daughters into a record-typed node (Java ``resolveRestrictor``)."""
        field_at = self.ttr_fields[node]
        if field_at is None or field_at.get_type() is None:
            raise LambdaTTRConversionError(f"restrictor node {node} has no record type")
        restrictor = TTRRecordType.parse(str(field_at.get_type()))
        if restrictor is None:
            raise LambdaTTRConversionError(
                f"could not parse restrictor {field_at.get_type()!s}"
            )
        for i in self.get_daughters(node):
            daughter = self.ttr_fields[i]
            if daughter is None:
                continue
            restrictor.add(daughter)
            self.ttr_fields[i] = None
        merged = TTRField.parse(f"{field_at.get_label()}:{restrictor}")
        if merged is None:
            raise LambdaTTRConversionError(f"merged restrictor did not parse: {restrictor}")
        self.ttr_fields[node] = merged
        self.resolveds[node] = True

    def resolve_event_restrictor(self, event_restr_node: int) -> None:
        """Resolve the event node once its daughters (or the rest of the graph) are done."""
        if self.new_node == -1:
            return
        current = self.ttr_fields[event_restr_node]
        if self.get_daughters(event_restr_node):
            if self.all_daughters_resolved(event_restr_node):
                if current is not None and str(current.get_label()).startswith("r"):
                    self.resolve_restrictor(event_restr_node)
                else:
                    self.resolveds[event_restr_node] = True
                return
        for i in range(self.new_node + 1):
            if i == event_restr_node:
                continue
            if not self.resolveds[i]:
                return
        self.resolveds[event_restr_node] = True

    def resolved_ttrs(self) -> list[TTRField]:
        """Collect resolved fields, children before parents (Java ``resolvedTTRs``)."""
        ttrs: list[TTRField] = []
        for j in range(self.new_node, -1, -1):
            if self.ttr_fields[j] is None and self.resolveds[j]:
                continue
            if self.resolveds[j]:
                field_at = self.ttr_fields[j]
                if field_at is not None:
                    ttrs.append(field_at)
            elif self.all_daughters_resolved(j) and self.ttr_fields[j] is not None:
                self.resolveds[j] = True
                field_at = self.ttr_fields[j]
                if field_at is not None:
                    ttrs.append(field_at)
        return ttrs


def sort_fields(fields: list[TTRField], head_move: bool) -> str:
    """Dependency-sort fields and render a record string (Java ``CorpusConverterAgenda.sort``)."""
    swapped = True
    while swapped:
        swapped = False
        i = 0
        while i < len(fields) - 1:
            field_at = fields[i]
            if not field_at.has_manifest():
                if str(field_at.get_label()).lower() == "head" and head_move:
                    fields.insert(i + 2, field_at)
                    del fields[i]
                    swapped = True
                else:
                    i += 1
                    continue
            else:
                manifest = field_at.get_type()
                body_src = "" if manifest is None else str(manifest)
                open_at = body_src.find("(")
                body = body_src[open_at + 1 : len(body_src) - 1] if open_at >= 0 else body_src
                arguments: list[str] = []
                arg = ""
                for ch in body:
                    if ch == ",":
                        arguments.append(arg)
                        arg = ""
                    elif ch != " ":
                        arg += ch
                arguments.append(arg)
                dependent = i
                for j in range(i + 1, len(fields)):
                    for myarg in arguments:
                        label = str(fields[j].get_label())
                        if label == myarg or (
                            "." in myarg and label == myarg[: myarg.index(".")]
                        ):
                            dependent = j + 1
                            swapped = True
                if swapped:
                    fields.insert(dependent, field_at)
                    del fields[i]
                    break
            i += 1
    parts: list[str] = []
    for index, field_at in enumerate(fields):
        parts.append(str(field_at))
        parts.append("]" if index == len(fields) - 1 else "|")
    return "[" + "".join(parts)


def _conjunct_split(my_string: str) -> list[str]:
    """Split a conjunction body on top-level commas (Java ``conjunctSplit``)."""
    conjuncts: list[str] = []
    popped = 0
    split_point = 0
    for c, ch in enumerate(my_string):
        if ch == "(":
            popped += 1
        elif ch == ")":
            popped -= 1
        if ch == "," and popped == 0:
            conjuncts.append(my_string[split_point:c])
            split_point = c + 1
        elif c == len(my_string) - 1:
            conjuncts.append(my_string[split_point:])
    if popped != 0:
        raise LambdaTTRConversionError(
            f"unbalanced brackets in conjunction: {my_string!r} ({popped})"
        )
    return conjuncts


class LambdaTTRConverter:
    """Rewrite one Eve-style lambda formula into a TTR record-type string."""

    def __init__(self) -> None:
        """Start with empty variable pools and no agenda."""
        self.p = -1
        self.x = -1
        self.ev = -1
        self.r = -1
        self.utterance = ""
        self.variables: dict[str, str] = {}
        self.event_var = ""
        self.internal_event = ""
        self.subject: list[str | None] = [None, None]
        self.head = ""
        self.event_restrictor_node = 1000
        self.control_verb = False
        self.agenda = ConverterAgenda()

    def fresh_var(self, type_name: str, new: bool) -> str:
        """Return the next ``p`` / ``e`` / ``x`` / ``r`` name (Java ``freshVar``)."""
        new_var: str | None = None
        kind = type_name.lower()
        if kind in {"p", "t"}:
            if new:
                self.p += 1
            new_var = "p" if self.p == 0 else f"p{self.p}"
        elif kind in {"ev", "es"}:
            if new:
                self.ev += 1
            new_var = "e" if self.ev == 0 else f"e{self.ev}"
        elif kind in {"x", "e"}:
            if new:
                self.x += 1
            new_var = "x" if self.x == 0 else f"x{self.x}"
        elif kind == "r":
            if new:
                self.r += 1
            new_var = "r" if self.r == 0 else f"r{self.r}"
        if new_var is None:
            raise LambdaTTRConversionError(f"no fresh variable for type {type_name}")
        return new_var

    def convert(self, orig_childes: str, utterance: str) -> str:
        """Convert one lambda formula paired with its utterance (Java ``TTRconvert``)."""
        self.__init__()
        self.utterance = utterance
        childes = orig_childes.strip()
        self.internal_event = self.fresh_var("ev", True)
        if childes.startswith("lambda"):
            parts = childes.split(".")
            for sub in parts[:-1]:
                match = _full(_LAMBDA, sub)
                if match is None:
                    continue
                variable = match.group(2).strip()
                type_name = match.group(4).strip()
                self.variables[variable] = self.fresh_var(type_name, True)
                if type_name.lower() == "e":
                    eventish = self.fresh_var("e", False)
                    self.variables[""] = (
                        f"{self.fresh_var('p', True)}==question_feature({eventish})"
                    )
            childes = parts[-1]
        for match in _VARIABLE_INIT.finditer(childes):
            variable = match.group(2).strip()
            if variable in self.variables:
                continue
            self.variables[variable] = self.fresh_var("e", True)
        self.head = ""
        for var, value in list(self.variables.items()):
            if value.startswith("e"):
                if self.event_var != "":
                    raise LambdaTTRConversionError(
                        f"more than one event variable in {orig_childes}"
                    )
                self.event_var = var
        if self.event_var == "":
            if _full(_DET_QUANT, childes) or _full(_ATOMIC_POS, childes):
                logger.debug("atomic or det-headed utterance: %s", childes)
            else:
                self.event_var = "$dummy"
                self.variables[self.event_var] = self.fresh_var("ev", True)
                self.internal_event = self.variables[self.event_var]
                self.event_restrictor_node = self.agenda.add_to_agenda(
                    orig_childes, f"{self.internal_event}:es", self.agenda.new_node
                )
                self.head = self.internal_event
        else:
            self.internal_event = self.variables[self.event_var]
            self.event_restrictor_node = self.agenda.add_to_agenda(
                orig_childes, f"{self.internal_event}:es", self.agenda.new_node
            )
            self.head = self.internal_event
        steps = 0
        while not (len(childes) == 0 and self.agenda.is_complete()):
            steps += 1
            if steps > 400:
                raise LambdaTTRConversionError(
                    f"converter did not finish (missing rule) for {orig_childes}"
                )
            if childes != "":
                if self.event_restrictor_node == 1000:
                    self.agenda.add_to_agenda(
                        childes, f"{self.fresh_var('x', True)}:e", self.agenda.new_node
                    )
                    self.head = self.fresh_var("x", False)
                else:
                    self.agenda.add_to_agenda(
                        childes,
                        f"{self.fresh_var('p', True)}:t",
                        self.agenda.parent_node_numbers[self.event_restrictor_node],
                    )
                    self.head = self.internal_event
                childes = ""
                continue
            for i in range(self.agenda.new_node, -1, -1):
                my_string = self.agenda.childes_strings[i]
                if my_string is None:
                    my_string = ""
                if not self.agenda.resolved(i):
                    if i == self.event_restrictor_node:
                        break
                    field_at = self.agenda.ttr_fields[i]
                    daughters_ready = (
                        field_at is not None
                        and self.agenda.all_daughters_resolved(i)
                        and field_at.get_type() is not None
                    )
                    restrictor_label = field_at is not None and str(field_at.get_label()).startswith(
                        "r"
                    )
                    if daughters_ready or restrictor_label:
                        if restrictor_label:
                            self.agenda.resolve_restrictor(i)
                        self.agenda.make_resolved(i)
                        continue
                else:
                    continue
                if my_string.startswith("Q(") or my_string.startswith("not("):
                    self._question_negation(my_string, i)
                    break
                if my_string.startswith("eq"):
                    self._eq_eq_loc(my_string, i)
                    break
                tense = _full(_TENSE_PREDICATE, my_string)
                if tense:
                    pred = tense.group(1)
                    pred_head = self.fresh_var("p", True)
                    self.agenda.make_resolved(
                        self.agenda.add_to_agenda(
                            my_string,
                            f"{pred_head}=={pred}_feature({self.internal_event}) : t",
                            self.event_restrictor_node,
                        )
                    )
                    self.agenda.childes_strings[i] = tense.group(3)
                    break
                conj = _full(_CONJUNCT_PREDICATE, my_string)
                if conj:
                    self._and(conj.group(2), i)
                    break
                det = _full(_DET_QUANT, my_string)
                if (
                    det
                    and "v" not in det.group(1)
                    and "part" not in det.group(1)
                    and "aux" not in det.group(1)
                    and "adv" not in det.group(1)
                    and "prep" not in det.group(1)
                ):
                    self._det_quant(my_string, i)
                    break
                pos = _full(_PREDICATE_POS, my_string)
                multi = _full(_PREDICATE_POS_MULTI, my_string)
                if pos or multi:
                    self._pos_pred(my_string, i)
                    break
                atomic = _full(_ATOMIC_POS, my_string)
                if atomic:
                    xish = self.fresh_var("e", True)
                    if field_at is not None and str(field_at.get_label()).startswith("x"):
                        xish = str(field_at.get_label())
                    literal = f"{xish}=={atomic.group(3).lower()}:e"
                    self.agenda.put_ttr_string(literal, i)
                    self.agenda.make_resolved(i)
                    break
                if my_string.startswith("$"):
                    if my_string in self.variables:
                        self.agenda.put_ttr_string("", i)
                        self.agenda.make_resolved(i)
                        break
                    raise LambdaTTRConversionError(f"no matching variable in {my_string}")
                if my_string.startswith("n|+n"):
                    self._nominal_compound(my_string, i)
                    break
            if (
                self.event_restrictor_node != 1000
                and self.agenda.ttr_fields[self.event_restrictor_node] is not None
            ):
                self.agenda.resolve_event_restrictor(self.event_restrictor_node)
        return self._finalise(orig_childes)

    def _question_negation(self, my_string: str, i: int) -> str:
        """Rewrite ``Q(…)`` / ``not(…)`` (Java ``questionNegation``)."""
        pred_name = "question_feature" if my_string.startswith("Q") else "not_feature"
        pred = f"{self.fresh_var('p', True)}=={pred_name}("
        match = _full(_FINAL_VARIABLE, my_string)
        if match:
            my_var = my_string[my_string.rindex("$") : len(my_string) - 1]
            if my_var.lower() != self.event_var.lower():
                raise LambdaTTRConversionError(
                    f"final variable is not the event variable: {my_string}"
                )
        else:
            raise LambdaTTRConversionError(f"not well formed Q/NOT: {my_string}")
        head_type = "es"
        internal = my_string[my_string.index("(") + 1 : my_string.rindex(",$")]
        if self.event_var not in internal:
            pos = _full(_PREDICATE_POS, internal)
            if pos:
                if pos.group(1).startswith("adj"):
                    self.head = self.internal_event
                    head_type = "es"
                elif _full(_DET_QUANT, internal):
                    self.head = self.fresh_var("x", True)
                    head_type = "e"
                else:
                    self.head = self.internal_event
                    head_type = "es"
            else:
                if not self.head.startswith("x"):
                    self.head = self.fresh_var("e", True)
                head_type = "e"
            if head_type == "e":
                for var in list(self.variables):
                    if var == self.event_var:
                        del self.variables[var]
                    self.agenda.remove_node(self.event_restrictor_node)
        else:
            if internal.startswith("Q()") or internal.startswith("not("):
                internal = self._question_negation(internal, i)
            else:
                self.head = self.internal_event
                head_type = "es"
        pred += f"{self.head}) : t"
        if self.head.lower() == self.internal_event.lower():
            self.agenda.make_resolved(
                self.agenda.add_to_agenda(my_string, pred, self.event_restrictor_node)
            )
        else:
            self.agenda.make_resolved(self.agenda.add_to_agenda(my_string, pred, i))
            self.agenda.put_ttr_string(f"{self.head}:{head_type}", i)
        self.agenda.childes_strings[i] = internal
        return internal

    def _eq_eq_loc(self, my_string: str, i: int) -> None:
        """Rewrite ``eq`` / ``eqLoc`` (Java ``eqEqLoc``, second version, no argument swap block)."""
        match = _full(_FINAL_VARIABLE, my_string)
        if not match:
            raise LambdaTTRConversionError(f"no final variable for eq/eqLoc: {my_string}")
        final_var = my_string[my_string.rindex("$") : len(my_string) - 1]
        if final_var == self.event_var:
            my_string = my_string[: my_string.rindex(",$")]
        elif final_var in self.variables:
            my_string = my_string[: len(my_string) - 1]
        else:
            raise LambdaTTRConversionError(f"no final variable in {my_string}")
        pred = my_string[: my_string.index("(")].lower()
        my_string = my_string[my_string.index("(") + 1 :]
        event_label = self.variables.get(self.event_var)
        if event_label is None:
            raise LambdaTTRConversionError("eq without an event variable")
        main_pred = f"{event_label}=={pred}"
        conjuncts = _conjunct_split(my_string)
        if len(conjuncts) != 2:
            raise LambdaTTRConversionError(f"irregular eq size {len(conjuncts)}")
        second_version = True
        if second_version:
            main_pred += ":es"
            self.agenda.remove_node(self.event_restrictor_node)
        if conjuncts[1] in self.variables:
            utt = self.utterance
            if pred.lower() == "eqloc" or not (
                utt.endswith("what") or utt.endswith("who") or utt.endswith("where")
            ):
                conjuncts.insert(0, conjuncts[1])
                del conjuncts[2]
        for c, conj in enumerate(conjuncts):
            var_bool = conj in self.variables
            conj1 = self.variables[conj] if var_bool else self.fresh_var("e", True)
            arg_type = "subj" if c == 0 else "obj"
            if second_version:
                arg_node = self.agenda.add_to_agenda(conj, "", i)
                my_node = self.agenda.add_to_agenda(conj, f"{conj1}:e", arg_node)
            else:
                my_node = self.agenda.add_to_agenda(conj, f"{conj1}:e", i)
                arg_node = 0
            if var_bool:
                self.agenda.put_ttr_string(None, my_node)
                self.agenda.make_resolved(my_node)
            if second_version:
                event_label = self.variables.get(self.event_var, "")
                self.agenda.put_ttr_string(
                    f"{self.fresh_var('p', True)}=={arg_type}({event_label},{conj1}):t",
                    arg_node,
                )
        self.agenda.put_ttr_string(main_pred, i)

    def _and(self, my_string: str, i: int) -> None:
        """Rewrite ``and`` conjuncts (Java ``and``)."""
        conjuncts = _conjunct_split(my_string)
        pred = ""
        nouns = False
        put = False
        gonna_conjunct = ""
        for c, conjunct in enumerate(conjuncts):
            pred_m = _full(_PREDICATE_POS_FINAL_VAR, conjunct)
            pred_m2 = _full(_TENSE_PREDICATE, conjunct)
            if pred_m or pred_m2:
                pos_type = ""
                pred_name = ""
                pred_body = ""
                if pred_m:
                    pos_type = pred_m.group(1)
                    pred_name = pred_m.group(3)
                    if pred_m.group(5) is not None:
                        pred_body = pred_m.group(5)
                elif pred_m2:
                    pred_name = pred_m2.group(1)
                    pred_body = pred_m2.group(3)
                if pred_name.startswith("put") or "|put" in pred_body:
                    put = True
                gonna_try = False
                if c == 0 and len(conjuncts) > 1 and (
                    conjuncts[1].startswith("v")
                    or (pos_type == "aux" and pred_body.startswith("and("))
                ):
                    gonna_try = True
                if gonna_try and gonna_conjunct == "" and c == 0:
                    gonna_conjunct = conjunct
                    gonna_try = False
                if c == 0:
                    if put:
                        self.agenda.childes_strings[i] = conjunct
                        break
                elif c == 1:
                    if gonna_conjunct != "":
                        self.agenda.childes_strings[i] = conjunct
                        self.agenda.add_to_agenda(
                            gonna_conjunct,
                            f"{self.fresh_var('es', True)}:es",
                            self.agenda.parent_node_numbers[i],
                        )
                        mapped = self.fresh_var("es", False)
                        self.variables[mapped] = mapped
                    elif pos_type.startswith("adv") or pos_type.startswith("prep"):
                        self.agenda.add_to_agenda(
                            conjunct,
                            f"{self.fresh_var('p', True)} : t",
                            self.event_restrictor_node,
                        )
                        self.agenda.childes_strings[i] = conjuncts[0]
                    else:
                        self.agenda.childes_strings[i] = conjuncts[0]
                        self.agenda.add_to_agenda(
                            conjunct,
                            f"{self.fresh_var('es', True)}:es",
                            self.agenda.parent_node_numbers[i],
                        )
                        mapped = self.fresh_var("es", False)
                        self.variables[mapped] = mapped
            else:
                if conjunct.startswith("n") or conjunct.startswith("pro"):
                    if not nouns:
                        current = self.agenda.ttr_fields[i]
                        pred = (
                            self.fresh_var("x", True)
                            if current is None
                            else str(current.get_label())
                        )
                        pred += "==and("
                        nouns = True
                    self.agenda.add_to_agenda(conjunct, f"{self.fresh_var('x', True)} : e", i)
                    pred += self.fresh_var("x", False)
                    pred = pred + "," if c < len(conjuncts) - 1 else pred + ") : e"
                elif conjunct in self.variables:
                    nouns = True
                    if c == 0:
                        current = self.agenda.ttr_fields[i]
                        pred = (
                            self.fresh_var("p", True)
                            if current is None
                            else str(current.get_label())
                        )
                        pred += "==and("
                    pred += self.variables[conjunct]
                    pred = pred + "," if c < len(conjuncts) - 1 else pred + ") : t"
        if nouns:
            self.agenda.put_ttr_string(pred, i)
            return
        if put:
            self._put_oblique(conjuncts[1], i)

    def _put_oblique(self, my_string: str, i: int) -> None:
        """Rewrite the oblique conjunct of a ``put`` (Java ``and`` PUT branch)."""
        match = _full(_PREDICATE_POS, my_string)
        if not match:
            return
        event_label = self.variables.get(self.event_var)
        if event_label is None:
            raise LambdaTTRConversionError("put without an event variable")
        pred = f"{self.fresh_var('p', True)}==ind_obj({event_label},"
        pos_type = match.group(1)
        pred_name = match.group(3)
        body = match.group(5)
        restrictor_label = self.fresh_var("r", True)
        if pred_name.startswith("put"):
            return
        child = self.agenda.add_to_agenda(
            my_string,
            f"{self.fresh_var('es', True)}==epsilon({restrictor_label},{restrictor_label}.head):es",
            self.agenda.parent_node_numbers[i],
        )
        self.agenda.make_resolved(child)
        pred += f"{self.fresh_var('ev', False)}):t"
        child = self.agenda.add_to_agenda(my_string, pred, self.agenda.parent_node_numbers[i])
        self.agenda.make_resolved(child)
        put_rt = TTRRecordType()
        put_rt.add(self._must_field(f"{self.fresh_var('ev', True)}:es"))
        put_rt.add(self._must_field(f"head=={self.fresh_var('ev', False)}:es"))
        if ":" in pos_type:
            put_rt.add(
                self._must_field(
                    f"{self.fresh_var('p', True)}=={pos_type[pos_type.index(':') + 1]}"
                    f"({self.fresh_var('ev', False)}):t"
                )
            )
        pred = f"{self.fresh_var('p', True)}=={pred_name}({self.fresh_var('ev', False)}"
        conjuncts = _conjunct_split(body)
        if conjuncts and conjuncts[-1].lower() == self.event_var.lower():
            conjuncts.pop()
        else:
            raise LambdaTTRConversionError(
                f"no final event var {self.event_var} in {body}"
            )
        if not conjuncts:
            pred += "):t"
            put_rt.add(self._must_field(pred))
            self.agenda.make_resolved(
                self.agenda.add_to_agenda(
                    my_string,
                    f"{restrictor_label}:{put_rt}",
                    self.agenda.parent_node_numbers[i],
                )
            )
            return
        for c, conj in enumerate(conjuncts):
            if _full(_DET_QUANT, conj) or _full(_ATOMIC_POS, conj):
                pred += f",{self.fresh_var('e', True)}):t"
                put_rt.add(self._must_field(pred))
                my_restr = self.agenda.add_to_agenda(
                    my_string,
                    f"{restrictor_label}:{put_rt}",
                    self.agenda.parent_node_numbers[i],
                )
                self.agenda.add_to_agenda(conj, f"{self.fresh_var('e', False)}:e", my_restr)
            if c > 0:
                raise LambdaTTRConversionError(
                    f"more than 1 argument in prep phrase: {my_string}"
                )

    def _pos_pred(self, my_string: str, i: int) -> None:
        """Rewrite a POS-tagged predicate (Java ``POSpred``)."""
        match = _full(_PREDICATE_POS, my_string)
        multi = _full(_PREDICATE_POS_MULTI, my_string)
        pos_type = ""
        pred_name = ""
        body = ""
        if match:
            pos_type = match.group(1)
            pred_name = match.group(3)
            body = match.group(5)
        elif multi:
            pos_type = multi.group(1)
            pred_name = multi.group(3)
            body = multi.group(5)
            for found in _DUPLICATE_TYPES.finditer(pred_name):
                pred_name += found.group(4)
        pred = ""
        if pos_type.startswith(("v", "aux", "part", "adv", "prep")):
            pred += pred_name
            split_symbol = "-" if "-" in pred else "&" if "&" in pred else ""
            bound_event = self.internal_event
            if split_symbol:
                feature = my_string[my_string.index(split_symbol) + 1 : my_string.index("(")]
                ag = self.agenda.add_to_agenda(
                    my_string,
                    f"{self.fresh_var('p', True)}=={feature}_feature({bound_event}) : t",
                    self.event_restrictor_node,
                )
                self.agenda.make_resolved(ag)
                pred = pred[: pred.index(split_symbol)]
            if pos_type == "aux":
                pred = (
                    f"{self.fresh_var('p', True)}=={pred}_{pos_type}_feature({bound_event}):t"
                )
                ag2 = self.agenda.add_to_agenda(my_string, pred, self.event_restrictor_node)
                self.agenda.make_resolved(ag2)
                body += ")"
                final = _full(_FINAL_VARIABLE, body)
                if not final:
                    raise LambdaTTRConversionError(f"unknown pred in aux body: {body}")
                self.agenda.put_ttr_string(f"{self.fresh_var('t', True)}:t", i)
                self.agenda.childes_strings[i] = final.group(1)
                return
            conjuncts = _conjunct_split(body)
            if conjuncts and conjuncts[-1].lower() == self.event_var.lower():
                conjuncts.pop()
            else:
                raise LambdaTTRConversionError(
                    f"no final event var {self.event_var} in {body}"
                )
            event_pred = len(conjuncts) == 0
            if pos_type.startswith("adv") or pos_type.startswith("prep"):
                for value in self.variables.values():
                    if value.startswith("e") and value != self.internal_event:
                        bound_event = value
                self.agenda.parent_node_numbers[i] = self.event_restrictor_node
                if event_pred:
                    pred = f"{self.fresh_var('p', True)}=={pred}({bound_event}):t"
                    self.agenda.put_ttr_string(pred, i)
                    self.agenda.make_resolved(i)
                    return
                pred = f"{self.fresh_var('p', True)}=={pred}({bound_event},"
                for c, conj in enumerate(conjuncts):
                    if conj in self.variables and len(conjuncts) > 1:
                        raise LambdaTTRConversionError(
                            f"adjunct links out to a variable: {my_string}"
                        )
                    pred += self.fresh_var("e", True)
                    self.agenda.add_to_agenda(
                        conj,
                        f"{self.fresh_var('e', False)}:e",
                        self.event_restrictor_node,
                    )
                    pred += "," if c < len(conjuncts) - 1 else "):t"
                self.agenda.put_ttr_string(pred, i)
                self.agenda.make_resolved(i)
                return
            main_pred = f"=={pred}:es"
            bound_event = self.internal_event
            subject_copy = False
            for c, conj in enumerate(conjuncts):
                pred = f"{self.fresh_var('p', True)}=="
                if c == 0:
                    pred += "subj("
                    if self.subject[0] is None:
                        subject_copy = True
                    else:
                        self.control_verb = True
                        for value in self.variables.values():
                            if value.startswith("e") and value != self.internal_event:
                                bound_event = value
                        if bound_event == self.internal_event:
                            self.variables[self.fresh_var("es", True)] = self.fresh_var("es", False)
                            bound_event = self.fresh_var("es", False)
                elif c == 1:
                    pred += "obj("
                elif c == 2:
                    pred += "ind_obj("
                else:
                    raise LambdaTTRConversionError(f"{my_string} has over 3 arguments")
                pred += f"{bound_event},"
                if conj in self.variables and len(conjuncts) > 1:
                    pred += f"{self.variables[conj]}):t"
                    conj_int = self.agenda.add_to_agenda(conj, pred, i)
                    self.agenda.make_resolved(conj_int)
                else:
                    variable = False
                    if conj in self.variables:
                        e_arg = self.variables[conj]
                        variable = True
                    elif c == 0:
                        e_arg = (
                            self.fresh_var("e", True)
                            if self.subject[0] is None or self.subject[0] != conj
                            else str(self.subject[1])
                        )
                    else:
                        e_arg = self.fresh_var("e", True)
                    pred += f"{e_arg}):t"
                    conj_int = self.agenda.add_to_agenda(conj, pred, i)
                    if c == 0 and self.subject[0] is not None and conj == self.subject[0]:
                        self.agenda.make_resolved(conj_int)
                    elif not variable:
                        self.agenda.add_to_agenda(conj, f"{e_arg}:e", conj_int)
                    else:
                        self.agenda.make_resolved(conj_int)
                    if subject_copy:
                        self.subject[0] = conj
                        self.subject[1] = e_arg
                    subject_copy = False
            main_pred = bound_event + main_pred
            self.agenda.remove_node(self.event_restrictor_node)
            self.agenda.put_ttr_string(main_pred, i)
            self.agenda.make_resolved(i)
            return
        if match and (match.group(1).startswith("adj") or match.group(1).startswith("n")):
            if self.event_var == "":
                self.event_var = "$dummy"
                self.variables[self.event_var] = self.internal_event
            event_label = self.variables[self.event_var]
            if match.group(1).startswith("adj"):
                pred = f"{self.fresh_var('p', True)}=={match.group(3)}({event_label}):t"
            else:
                pred = f"{event_label}==eq:es"
                self.agenda.remove_node(self.event_restrictor_node)
                my_obj = self.agenda.add_to_agenda(
                    match.group(3),
                    f"{self.fresh_var('p', True)}==obj({event_label},{self.fresh_var('e', True)}):t",
                    i,
                )
                self.agenda.add_to_agenda(
                    f"{match.group(1)}|{match.group(3)}",
                    f"{self.fresh_var('e', False)}:e",
                    my_obj,
                )
            self.agenda.put_ttr_string(pred, i)
            if match.group(5) in self.variables:
                arg = self.variables[match.group(5)]
            else:
                arg = self.fresh_var("e", True)
                self.agenda.add_to_agenda(match.group(5), f"{arg}:e", i)
            subj_int = self.agenda.add_to_agenda(
                match.group(5),
                f"{self.fresh_var('p', True)}==subj({self.variables[self.event_var]},{arg}):t",
                i,
            )
            self.agenda.make_resolved(subj_int)
            self.head = self.internal_event
            return
        raise LambdaTTRConversionError(f"unknown predicate type in {my_string}")

    def _det_quant(self, my_string: str, i: int) -> None:
        """Rewrite a determiner or quantifier (Java ``detQuant``)."""
        match = _full(_DET_QUANT, my_string)
        if match is None:
            raise LambdaTTRConversionError(f"det/quant did not match: {my_string}")
        restrictor = f"{self.fresh_var('r', True)}:["
        current = self.agenda.ttr_fields[i]
        if current is None or str(current.get_label()).startswith("p"):
            det_ttr = f"{self.fresh_var('e', True)}=="
        else:
            det_ttr = f"{current.get_label()}=="
        word = match.group(3)
        if word.lower() == "a":
            quant = "epsilon"
        elif word.lower() == "the":
            quant = "iota"
        elif word.lower() in {"all", "every", "each"}:
            quant = "tau"
        else:
            quant = word
        det_ttr += (
            f"{quant}({self.fresh_var('r', False)}.head, {self.fresh_var('r', False)}) : e"
        )
        if match.group(5) in self.variables:
            my_variable = self.variables[match.group(5)]
        else:
            self.variables[match.group(5)] = self.fresh_var("e", True)
            my_variable = self.variables[match.group(5)]
        restrictor += f"{my_variable} :e|head=={my_variable}:e"
        del self.variables[match.group(5)]
        body = match.group(7)
        if body.startswith("and("):
            body = body[body.index("(") + 1 : len(body) - 1]
        for conjunct in body.split(","):
            multi = _full(_PREDICATE_POS_MULTI, conjunct)
            if multi:
                inner = multi.group(3)
                restrictor += f"|{self.fresh_var('p', True)}=="
                for found in _DUPLICATE_TYPES.finditer(inner):
                    restrictor += found.group(4)
                restrictor += f"({my_variable}) : t"
            else:
                pos = _full(_PREDICATE_POS, conjunct)
                if pos:
                    restrictor += (
                        f"|{self.fresh_var('p', True)}=={pos.group(3)}({my_variable}) : t"
                    )
        restrictor += "]"
        self.agenda.put_ttr_string(det_ttr.lower(), i)
        self.agenda.make_resolved(i)
        restr_int = self.agenda.add_to_agenda(
            body, restrictor.lower(), self.agenda.parent_node_numbers[i]
        )
        self.agenda.make_resolved(restr_int)

    def _nominal_compound(self, my_string: str, i: int) -> None:
        """Rewrite ``n|+n|…`` compounds (Java rule 9)."""
        match = _full(_PREDICATE_POS_MULTI_PRED, my_string)
        if not match:
            return
        body = match.group(3)
        arg = my_string[my_string.index("(") + 1 : my_string.rindex(")")]
        if self.event_var == "":
            self.event_var = "$dummy"
            self.variables[self.event_var] = self.internal_event
        pred_name = ""
        for found in _DUPLICATE_TYPES.finditer(body):
            pred_name += found.group(4)
        event_label = self.variables.get(self.event_var)
        if not event_label:
            raise LambdaTTRConversionError(
                f"nominal compound has no event variable left: {my_string}"
            )
        self.agenda.put_ttr_string(f"{event_label}==eq:es", i)
        self.agenda.make_resolved(i)
        self.agenda.remove_node(self.event_restrictor_node)
        my_obj = self.agenda.add_to_agenda(
            my_string,
            f"{self.fresh_var('p', True)}==obj({event_label},{self.fresh_var('e', True)}):t",
            self.agenda.parent_node_numbers[i],
        )
        self.agenda.make_resolved(
            self.agenda.add_to_agenda(
                my_string, f"{self.fresh_var('e', False)}=={pred_name}:e", my_obj
            )
        )
        my_subj = self.agenda.add_to_agenda(
            my_string,
            f"{self.fresh_var('p', True)}==subj({event_label},{self.fresh_var('e', True)}):t",
            self.agenda.parent_node_numbers[i],
        )
        self.agenda.add_to_agenda(arg, f"{self.fresh_var('e', False)}:e", my_subj)

    def _finalise(self, orig_childes: str) -> str:
        """Add variables, wh-words, head, and control-verb fixes (Java tail of ``TTRconvert``)."""
        ttrs = self.agenda.resolved_ttrs()
        for value in self.variables.values():
            if value.startswith("e"):
                continue
            if value.startswith("x"):
                parsed = TTRField.parse(f"{value}:e")
            elif value.startswith("p"):
                if "==" in value:
                    label = value[: value.index("=")]
                    type_s = value[value.index("==") + 2 :]
                    parsed = TTRField.parse(f"{label}=={type_s}:t")
                else:
                    parsed = TTRField.parse(f"{value}:t")
            else:
                parsed = None
            if parsed is not None:
                ttrs.append(parsed)
        head_type = "e" if self.head.startswith("x") else "es"
        words = self.utterance.split()
        wh_words = [word for word in words if word.lower() in {"what", "who", "where"}]
        t_length = 0
        while t_length < len(ttrs):
            my_size = len(ttrs)
            for f in range(my_size):
                field_at = ttrs[f]
                manifest = field_at.get_type()
                if manifest is not None and "_feature" in str(manifest):
                    text = str(manifest)
                    if (
                        text.startswith("question_")
                        or "zero" in text
                        or "one" in text
                        or "two" in text
                        or "three" in text
                    ):
                        del ttrs[f]
                        if text.startswith("question_") and isinstance(
                            manifest, PredicateArgumentFormula
                        ):
                            args = list(manifest.arguments)
                            if args and str(args[0]).startswith("e"):
                                break
                            t_index = 0
                            for myfield in list(ttrs):
                                if (
                                    str(myfield.get_label()) == str(args[0])
                                    and myfield.get_type() is None
                                ):
                                    if not wh_words:
                                        raise LambdaTTRConversionError(
                                            "question feature without a wh-word"
                                        )
                                    replacement = TTRField.parse(
                                        f"{myfield.get_label()}=={wh_words.pop(0)}:e"
                                    )
                                    if replacement is None:
                                        raise LambdaTTRConversionError("wh field did not parse")
                                    ttrs.append(replacement)
                                    del ttrs[t_index]
                                    break
                                t_index += 1
                        break
                    if "_aux_" not in text and "not_" not in text:
                        del ttrs[f]
                        break
                t_length = f + 1
        for f, myfield in enumerate(list(ttrs)):
            if myfield.get_type() is None and myfield.get_ds_type() is not None:
                if str(myfield.get_ds_type()) == "e":
                    if not wh_words:
                        break
                    replacement = TTRField.parse(
                        f"{myfield.get_label()}=={wh_words.pop(0)}:e"
                    )
                    if replacement is None:
                        raise LambdaTTRConversionError("bare wh field did not parse")
                    del ttrs[f]
                    ttrs.append(replacement)
                    break
        lone_adverb = True
        for myfield in ttrs:
            manifest = myfield.get_type()
            if manifest is not None and str(manifest).startswith("subj("):
                lone_adverb = False
                break
        if self.head.startswith("x"):
            lone_adverb = False
        if lone_adverb:
            rewritten: list[TTRField] = []
            for myfield in ttrs:
                text = str(myfield)
                if self.head and self.head in text:
                    text = text.replace(self.head, "head")
                parsed = TTRField.parse(text)
                if parsed is None:
                    raise LambdaTTRConversionError(f"lone-adverb field did not parse: {text}")
                rewritten.append(parsed)
            ttrs = rewritten
            rendered = sort_fields(ttrs, False)
        else:
            head_field = TTRField.parse(f"head=={self.head}:{head_type}")
            if head_field is None:
                raise LambdaTTRConversionError(f"head field did not parse: {self.head}")
            ttrs.append(head_field)
            if self.control_verb:
                ttrs = self._rewrite_control(ttrs)
            rendered = sort_fields(ttrs, True)
        if TTRRecordType.parse(rendered) is None:
            raise LambdaTTRConversionError(
                f"result is not valid TTR: {rendered} from {orig_childes}"
            )
        return rendered

    def _rewrite_control(self, ttrs: list[TTRField]) -> list[TTRField]:
        """Point control-verb fields at the embedded event (Java control-verb block)."""
        new_head = ""
        for afield in ttrs:
            label = str(afield.get_label())
            if label.startswith("e") and label != self.head:
                new_head = label
        if new_head == "":
            raise LambdaTTRConversionError("control verb has no new head")
        subject = ""
        remove: list[int] = []
        additions: list[TTRField] = []
        for f, myfield in enumerate(ttrs):
            manifest = myfield.get_type()
            if manifest is None:
                continue
            if str(manifest).startswith("subj") and isinstance(manifest, PredicateArgumentFormula):
                args = list(manifest.arguments)
                test = str(args[1]) if len(args) > 1 else ""
                if subject == "":
                    subject = test
                elif subject != test:
                    raise LambdaTTRConversionError("object-control subjects differ")
                elif str(args[0]) == self.head:
                    remove.append(f)
            elif str(myfield.get_label()) == "head":
                remove.append(f)
                parsed = TTRField.parse(f"head=={new_head}:es")
                if parsed is None:
                    raise LambdaTTRConversionError("control head did not parse")
                additions.append(parsed)
            elif str(myfield.get_label()) == self.head:
                remove.append(f)
                parsed = TTRField.parse(
                    f"{self.fresh_var('p', True)}=={manifest}({new_head}):t"
                )
                if parsed is None:
                    raise LambdaTTRConversionError("control predicate did not parse")
                additions.append(parsed)
            elif isinstance(manifest, PredicateArgumentFormula):
                if any(str(arg) == self.head for arg in manifest.arguments):
                    new_args = [
                        new_head if str(arg) == self.head else str(arg)
                        for arg in manifest.arguments
                    ]
                    new_pred = f"{manifest.predicate}(" + ",".join(new_args) + ")"
                    ds = myfield.get_ds_type()
                    parsed = TTRField.parse(
                        f"{myfield.get_label()}=={new_pred}:{ds}"
                    )
                    if parsed is None:
                        raise LambdaTTRConversionError("rewritten control field did not parse")
                    remove.append(f)
                    additions.append(parsed)
        shift = 0
        for index in remove:
            del ttrs[index - shift]
            shift += 1
        ttrs.extend(additions)
        return ttrs

    @staticmethod
    def _must_field(text: str) -> TTRField:
        """Parse *text* or raise when it is not a TTR field."""
        parsed = TTRField.parse(text)
        if parsed is None:
            raise LambdaTTRConversionError(f"field did not parse: {text}")
        return parsed


def convert_lambda(semantics: str, utterance: str) -> str:
    """Convert one lambda-calculus CHILDES formula to a TTR record-type string.

    :param semantics: Eve-style formula, for example ``lambda $0_{ev}.v|go(pro|you,$0)``.
    :param utterance: Surface string used for wh-word placement.
    :returns: A TTR record-type string.
    :raises LambdaTTRConversionError: When the existing rules do not cover *semantics*.
    """
    return LambdaTTRConverter().convert(semantics, utterance)


@dataclass(frozen=True)
class LambdaPair:
    """One utterance/formula pair from a ``trainPairs`` file."""

    file_index: int
    utterance: str
    semantics: str
    commented: bool


def iter_train_pairs(folder: str | Path) -> list[LambdaPair]:
    """Read ``trainPairs_1`` … ``trainPairs_20`` (Java ``CorpusReaderWriter`` layout).

    Commented ``//example_end`` blocks are returned with ``commented=True``.
    """
    root = Path(folder)
    pairs: list[LambdaPair] = []
    for index in range(1, 21):
        path = root / f"trainPairs_{index}"
        if not path.exists():
            continue
        sent: str | None = None
        sem: str | None = None
        commented = False
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("//Sent:"):
                sent = line.split(":", 1)[1].strip()
                commented = True
            elif line.startswith("Sent:"):
                sent = line.split(":", 1)[1].strip()
                commented = False
            elif line.startswith("//Sem:"):
                sem = line.split(":", 1)[1].strip()
                commented = True
            elif line.startswith("Sem:"):
                sem = line.split(":", 1)[1].strip()
            elif line.startswith("//example_end"):
                if sent is not None and sem is not None:
                    pairs.append(LambdaPair(index, sent, sem, True))
                sent = sem = None
                commented = False
            elif line.startswith("example_end"):
                if sent is not None and sem is not None:
                    pairs.append(LambdaPair(index, sent, sem, commented))
                sent = sem = None
                commented = False
    return pairs


@dataclass
class ConversionReport:
    """Counts from a batch lambda-to-TTR run."""

    source_pairs: int
    commented: int
    converted: int
    failed: int
    failures: list[tuple[str, str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """One-line count summary."""
        return (
            f"source={self.source_pairs} commented={self.commented} "
            f"converted={self.converted} failed={self.failed}"
        )


def convert_train_pair_folder(
    folder: str | Path,
    target: str | Path | None = None,
    *,
    include_commented: bool = False,
    failure_limit: int = 40,
) -> ConversionReport:
    """Convert every lambda pair under *folder* and optionally write TTR blocks.

    :param folder: Directory containing ``trainPairs_*``.
    :param target: Output path for ``Sent`` / ``Sem`` / ``File`` blocks.
    :param include_commented: Also attempt pairs closed by ``//example_end``.
    :param failure_limit: How many failure triples to keep on the report.
    """
    pairs = iter_train_pairs(folder)
    report = ConversionReport(
        source_pairs=len(pairs),
        commented=sum(1 for pair in pairs if pair.commented),
        converted=0,
        failed=0,
    )
    lines: list[str] = []
    for pair in pairs:
        if pair.commented and not include_commented:
            continue
        try:
            ttr = convert_lambda(pair.semantics, pair.utterance.rstrip(" .?"))
        except LambdaTTRConversionError as exc:
            report.failed += 1
            if len(report.failures) < failure_limit:
                report.failures.append((pair.utterance, pair.semantics, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 — batch must record unexpected rule gaps
            report.failed += 1
            if len(report.failures) < failure_limit:
                report.failures.append((pair.utterance, pair.semantics, f"{type(exc).__name__}: {exc}"))
            continue
        report.converted += 1
        lines.append(f"Sent : {pair.utterance}\nSem : {ttr}\nFile : {pair.file_index}\n")
    if target is not None:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
    return report
