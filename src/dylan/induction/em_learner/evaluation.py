"""Evaluation helpers for induction outputs (Java ``qmul.ds.learn.Evaluation``).

Literal port of Julian/AA's maximal-mapping precision/recall over hypothesised vs
gold TTR record-type pairs, including macro/micro averaging and best-interpretation
selection with mutual-subsumption short-circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from loguru import logger

from dylan.formula.predicate_argument import PredicateArgumentFormula
from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_label import TTRLabel
from dylan.formula.ttr_path import TTRPath, TTRRelativePath, parse_ttr_path
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Java inner class ``EvaluationResult``: precision/recall/F-score holder."""

    precision: float
    recall: float
    f_score: float

    @classmethod
    def from_counts(cls, hyp_total: float, goal_total: float, nodes_mapped: float) -> EvaluationResult:
        """Build from node totals (Java ``EvaluationResult(hypTotal, goalTotal, nodesMapped)``)."""
        recall = 0.0 if nodes_mapped == 0 else float(nodes_mapped) / float(goal_total)
        precision = 0.0 if nodes_mapped == 0 else float(nodes_mapped) / float(hyp_total)
        if recall == 0 or precision == 0:
            f_score = 0.0
        else:
            f_score = (2.0 * recall * precision) / (recall + precision)
        return cls(precision, recall, f_score)

    def get_precision(self) -> float:
        """Java ``getPrecision``."""
        return self.precision

    def get_recall(self) -> float:
        """Java ``getRecall``."""
        return self.recall

    def get_f_score(self) -> float:
        """Java ``getFScore``."""
        return self.f_score


class Evaluation:
    """Java ``Evaluation``: precision/recall over predicted vs gold TTR pairs."""

    def remove_label(self, t: TTRRecordType, label: TTRLabel) -> TTRRecordType:
        """Return a copy of *t* without field *label* (Java ``removeLabel``)."""
        result = TTRRecordType()
        for f in t.get_fields():
            if f.get_label() != label:
                result.add(f)
        return result

    def order_by_manifest(self, t: TTRRecordType, m: bool) -> TTRRecordType:
        """Reorder fields giving precedence to manifest / pred-arg / epsilon (Java ``orderByManifest``)."""
        result = TTRRecordType()
        manifest: list[TTRField] = []
        unmanifest: list[TTRField] = []
        pred_arg: list[TTRField] = []
        epsilons: list[TTRField] = []
        for my_field in t.get_fields():
            if my_field.get_type() is None or my_field.get_ds_type() is None:
                unmanifest.append(my_field)
            elif (
                isinstance(my_field.get_type(), PredicateArgumentFormula)
                or (str(my_field.get_label()) == "head" and my_field.get_type() is not None)
            ):
                typ = my_field.get_type()
                if isinstance(typ, PredicateArgumentFormula) and "." in str(typ):
                    epsilons.append(my_field)
                else:
                    pred_arg.append(my_field)
            else:
                manifest.append(my_field)
        if m:
            unmanifest.extend(manifest)
            unmanifest.extend(epsilons)
            unmanifest.extend(pred_arg)
            logger.debug("{}", unmanifest)
            for a in range(len(unmanifest) - 1, -1, -1):
                field = unmanifest[a]
                logger.debug("adding {}", field)
                if field.get_ds_type() is None:
                    result.add_at_top(field.get_label(), field.get_type(), None)  # type: ignore[arg-type]
                elif field.get_type() is None:
                    result.add_at_top(field.get_label(), None, field.get_ds_type())  # type: ignore[arg-type]
                else:
                    result.add_at_top(field.get_label(), field.get_type(), field.get_ds_type())  # type: ignore[arg-type]
                logger.debug("{}", field)
        else:
            manifest.extend(unmanifest)
            for field in manifest:
                result.add(field)
        logger.debug("------")
        return result

    def maximal_mapping(
        self,
        hypttr: TTRRecordType,
        o: TTRRecordType,
        map_: dict[Variable, Variable],
    ) -> dict[Variable, Variable]:
        """Exhaustive field mapping from *hypttr* into *o* (Java ``maximalMapping``)."""
        logger.debug("TOP LEVEL checking max mapping {} is subsumed by {}", hypttr, o)
        if hypttr.is_empty():
            logger.debug("final mapping")
            for v in map_:
                logger.debug("{}:{}", v, map_[v])
            return map_
        if not isinstance(o, TTRRecordType):
            return map_

        other = o
        logger.debug("{}", hypttr)
        logger.debug("{}", other)
        other = self.order_by_manifest(other, True)
        hypttr = self.order_by_manifest(hypttr, True)
        logger.debug("{}", hypttr)
        logger.debug("{}", other)
        last = hypttr.get_fields()[-1]
        logger.debug("OUR OUTER testing subsumption for field:{}", last)
        for j in range(len(other.get_fields()) - 1, -1, -1):
            other_field = other.get_fields()[j]
            logger.debug("inner checking {}", other_field)
            last_type = last.get_type()
            other_type = other_field.get_type()
            if (
                last_type is not None
                and other_type is not None
                and isinstance(last_type, TTRRecordType)
                and isinstance(other_type, TTRRecordType)
            ):
                logger.debug("{} and {} both RT types", last, other_field)
                embeddedmap = self.maximal_mapping(last_type, other_type, {})
                if not embeddedmap or (
                    str(last_type) == "[]" and str(other_type) == "[]"
                ):
                    logger.debug("{} internal subsumes {}", last, other_field)
                    map_[last.get_label()] = other_field.get_label()  # type: ignore[index,assignment]
                    logger.debug("map is now:{}", map_)
                    return self.maximal_mapping(
                        self.remove_label(hypttr, last.get_label()),  # type: ignore[arg-type]
                        self.remove_label(other, other_field.get_label()),  # type: ignore[arg-type]
                        map_,
                    )
            elif (
                (last_type is None or not isinstance(last_type, PredicateArgumentFormula))
                and (other_type is None or not isinstance(other_type, PredicateArgumentFormula))
                and last.subsumes_mapped(other_field, map_)
            ):
                logger.debug("{} subsumes {}", last, other_field)
                logger.debug("map is now:{}", map_)
                return self.maximal_mapping(
                    self.remove_label(hypttr, last.get_label()),  # type: ignore[arg-type]
                    self.remove_label(other, other_field.get_label()),  # type: ignore[arg-type]
                    map_,
                )
            else:
                logger.debug("checking partial subsumption for {} against {}", last, other_field)
                partially_subsumes_mapped = False
                if last.get_label().subsumes_mapped(other_field.get_label(), map_) and (
                    (last.get_ds_type() is None and other_field.get_ds_type() is None)
                    or (
                        last.get_ds_type() is not None
                        and last.get_ds_type() == other_field.get_ds_type()
                    )
                ):
                    logger.debug("field labels subsume {}{}", last, other_field)
                    if last_type is not None and other_type is not None:
                        if isinstance(last_type, PredicateArgumentFormula) and isinstance(
                            other_type, PredicateArgumentFormula
                        ):
                            logger.debug("both pred types {}{}", last, other_field)
                            if last_type.predicate == other_type.predicate:
                                partially_subsumes_mapped = True
                                logger.debug("pred match!{}{}", last, other_field)
                                k = 0
                                myargs = list(last_type.arguments)
                                otherargs = list(other_type.arguments)
                                if len(myargs) > len(otherargs):
                                    myargs = myargs[: len(otherargs)]
                                for i in range(len(myargs)):
                                    logger.debug("arg{}  {}", i, myargs[i])
                                    logger.debug("{}", hypttr)
                                    logger.debug("{}", other)
                                    my_s = str(myargs[i])
                                    other_s = str(otherargs[i])
                                    if ((".") in my_s or my_s.startswith("r")) and (
                                        (".") in other_s or other_s.startswith("r")
                                    ):
                                        logger.debug("both paths or restrictors")
                                        my_path_string = my_s if "." in my_s else "." + my_s
                                        other_path_string = (
                                            other_s if "." in other_s else "." + other_s
                                        )
                                        mypath = parse_ttr_path(my_path_string)
                                        other_path = parse_ttr_path(other_path_string)
                                        if not isinstance(mypath, TTRRelativePath) or not isinstance(
                                            other_path, TTRRelativePath
                                        ):
                                            continue
                                        logger.debug(
                                            "both now paths  {} and {}",
                                            my_path_string,
                                            other_path_string,
                                        )
                                        mypath.parent_rec_type = hypttr
                                        other_path.parent_rec_type = other
                                        logger.debug("mypath{}", mypath.get_minimal_super_type_with())
                                        try:
                                            logger.debug(
                                                "otherpath{}",
                                                other_path.get_minimal_super_type_with(),
                                            )
                                        except Exception:  # noqa: BLE001
                                            logger.error(
                                                "already removed  restrictor as it doesn't match.."
                                            )
                                            continue
                                        if len(mypath.get_labels()) > len(
                                            other_path.get_labels()
                                        ) or not other_path.get_minimal_super_type_with().subsumes(
                                            mypath.get_minimal_super_type_with()
                                        ):
                                            continue
                                        k += 1
                                    else:
                                        hyp_rec = hypttr.get_record()
                                        other_rec = other.get_record()
                                        my_lab = TTRLabel(str(myargs[i]))
                                        other_lab = TTRLabel(str(otherargs[i]))
                                        hyp_f = hyp_rec.get(my_lab)
                                        other_f = other_rec.get(other_lab)
                                        if hyp_f is not None and other_f is not None and hyp_f.subsumes(other_f):
                                            logger.debug("{}", hyp_f)
                                            logger.debug("{}", other_f)
                                            logger.debug("{}", map_)
                                            k += 1
                                            break

                                if k == -1:
                                    partially_subsumes_mapped = False
                                    map_.pop(last.get_label(), None)  # type: ignore[arg-type]
                            else:
                                partially_subsumes_mapped = False
                                logger.debug("Failed partial subsumption for: {}", other_field)
                                logger.debug("map is now:{}", map_)
                        elif isinstance(last_type, TTRRecordType) and isinstance(
                            other_type, TTRRecordType
                        ):
                            logger.debug("{} and {} both RT types", last, other_field)

                    if partially_subsumes_mapped:
                        logger.debug("Partially subsumed for :{}", other_field)
                        logger.debug("map is now:{}", map_)
                        return self.maximal_mapping(
                            self.remove_label(hypttr, last.get_label()),  # type: ignore[arg-type]
                            self.remove_label(other, other_field.get_label()),  # type: ignore[arg-type]
                            map_,
                        )
                    map_.pop(last.get_label(), None)  # type: ignore[arg-type]
                    logger.debug("Failed partial subsumption for: {}", other_field)
                    logger.debug("map is now:{}", map_)
                else:
                    map_.pop(last.get_label(), None)  # type: ignore[arg-type]
                    logger.debug("Failed subsumption for:{}", other_field)
                    logger.debug("map is now:{}", map_)

        hypttr = self.remove_label(hypttr, last.get_label())  # type: ignore[arg-type]
        return self.maximal_mapping(hypttr, o, map_)

    @staticmethod
    def field_total(ttr: TTRRecordType | None) -> int:
        """Count fields recursively, including embedded RTs (Java ``fieldTotal``)."""
        if ttr is None:
            return 0
        total = 0
        for f in ttr.get_fields():
            total += 1
            inner = f.get_type()
            if isinstance(inner, TTRRecordType):
                total += Evaluation.field_total(inner)
        return total

    def total_nodes_mapped(
        self,
        hypttr: TTRRecordType | None,
        goalttr: TTRRecordType | None,
    ) -> float:
        """Score mapped fields via maximal mapping (Java ``totalNodesMapped``)."""
        if hypttr is None or goalttr is None:
            return 0.0
        mapping = self.maximal_mapping(hypttr, goalttr, {})
        logger.debug("above is a mapping for \n{} and gold \n{}", hypttr, goalttr)

        field_score_map: dict[Variable, float] = {}
        total_mapped_nodes = 0.0

        for var in list(mapping.keys()):
            hyp_lab = TTRLabel(str(var))
            goal_lab = TTRLabel(str(mapping[var]))
            hyp_rec = hypttr.get_record()
            goal_rec = goalttr.get_record()
            if hyp_lab not in hyp_rec or goal_lab not in goal_rec:
                continue
            logger.debug("Var mapping= {}:{}", var, mapping[var])
            nodes_mapped = 0
            total_nodes = 0
            nodes_mapped += 1
            total_nodes += 1
            myfield = hyp_rec[hyp_lab]
            otherfield = goal_rec[goal_lab]
            my_type = myfield.get_type()
            other_type = otherfield.get_type()
            if my_type is not None:
                if isinstance(my_type, PredicateArgumentFormula) and isinstance(
                    other_type, PredicateArgumentFormula
                ):
                    total_nodes += 1
                    if my_type.predicate == other_type.predicate:
                        nodes_mapped += 1
                    else:
                        continue

                    args = list(my_type.arguments)
                    otherargs = list(other_type.arguments)
                    total_nodes += len(otherargs)
                    for a in range(len(args)):
                        if a >= len(otherargs):
                            continue
                        arg_s = str(args[a])
                        other_arg_s = str(otherargs[a])
                        if (
                            "." in arg_s
                            or (arg_s.startswith("r") and len(arg_s) < 3)
                        ) and (
                            "." in other_arg_s
                            or (other_arg_s.startswith("r") and len(other_arg_s) < 3)
                        ):
                            logger.debug("both paths or restrictors")
                            my_path_string = arg_s if "." in arg_s else "." + arg_s
                            other_path_string = (
                                other_arg_s if "." in other_arg_s else "." + other_arg_s
                            )
                            logger.debug(
                                "both now paths  {} and {}", my_path_string, other_path_string
                            )
                            mypath = parse_ttr_path(my_path_string)
                            other_path = parse_ttr_path(other_path_string)
                            if not isinstance(mypath, TTRRelativePath) or not isinstance(
                                other_path, TTRRelativePath
                            ):
                                continue
                            logger.debug("{}", hypttr)
                            logger.debug("{}", goalttr)
                            mypath.parent_rec_type = hypttr
                            other_path.parent_rec_type = goalttr
                            logger.debug("mypath{}", mypath.get_minimal_super_type_with())
                            logger.debug("otherpath{}", other_path.get_minimal_super_type_with())
                            total_nodes += len(other_path.get_labels()) - 1
                            if len(mypath.get_labels()) > len(
                                other_path.get_labels()
                            ) or not other_path.get_minimal_super_type_with().subsumes(
                                mypath.get_minimal_super_type_with()
                            ):
                                continue
                            nodes_mapped += 1
                            nodes_mapped += len(mypath.get_labels()) - 1
                        else:
                            my_arg = TTRLabel(arg_s)
                            other_arg = TTRLabel(other_arg_s)
                            if my_arg not in hyp_rec and other_arg not in goal_rec:
                                continue
                            hyp_af = hyp_rec.get(my_arg)
                            goal_af = goal_rec.get(other_arg)
                            if hyp_af is not None and goal_af is not None and hyp_af.subsumes(goal_af):
                                logger.debug("{}subsumes!", args[a])
                                nodes_mapped += 1
                elif isinstance(my_type, TTRRecordType) and isinstance(other_type, TTRRecordType):
                    logger.debug("part of embedded {} and {}", my_type, other_type)
                    nodes_mapped += int(self.total_nodes_mapped(my_type, other_type))
                    total_nodes += int(self.total_nodes_mapped(other_type, other_type))
                    total_mapped_nodes += self.total_nodes_mapped(my_type, other_type)
                else:
                    if myfield.get_label() == TTRLabel("head"):
                        hyphead = TTRLabel(str(my_type))
                        goalhead = (
                            otherfield.get_label()
                            if other_type is None
                            else TTRLabel(str(other_type))
                        )
                        hyp_hf = hypttr.get_record().get(hyphead)  # type: ignore[arg-type]
                        goal_hf = goalttr.get_record().get(goalhead)  # type: ignore[arg-type]
                        if hyp_hf is not None and goal_hf is not None and hyp_hf.subsumes(goal_hf):
                            nodes_mapped += 1
                    elif my_type == other_type:
                        nodes_mapped += 1
                    total_nodes += 1
            elif other_type is not None:
                total_nodes += 1
                if isinstance(other_type, PredicateArgumentFormula):
                    for v in other_type.arguments:
                        total_nodes += 1
                        if "." in str(v):
                            parsed = parse_ttr_path(str(v))
                            if isinstance(parsed, TTRRelativePath):
                                total_nodes += len(parsed.get_labels()) - 1
                            elif parsed is None:
                                # Java: TTRRelativePath.parse(v) — single-segment paths may be invalid
                                pass
            logger.debug("field total mapped {}", nodes_mapped)
            logger.debug("field total possible {}", total_nodes)
            if total_nodes:
                ratio = float(nodes_mapped) / float(total_nodes)
                total_mapped_nodes += ratio
                field_score_map[var] = ratio
        logger.debug("{}", field_score_map)
        logger.debug("{}", hypttr)
        logger.debug("{}", goalttr)
        return total_mapped_nodes

    def precision_recall(self, hypttr: TTRRecordType, goalttr: TTRRecordType) -> EvaluationResult:
        """Per-pair precision/recall via ``total_nodes_mapped`` denominators (Java ``precisionRecall``)."""
        logger.debug(
            "Calculating precision and recall for\n{}\nagainst gold \n{}\n",
            hypttr,
            goalttr,
        )
        total_nodes = self.total_nodes_mapped(hypttr, hypttr)
        goal_nodes = self.total_nodes_mapped(goalttr, goalttr)
        nodes_mapped = self.total_nodes_mapped(hypttr, goalttr)
        logger.debug("nodesMapped {}", nodes_mapped)
        logger.debug("totalNodes {}", total_nodes)
        logger.debug("goalNodes {}", goal_nodes)
        if total_nodes != float(Evaluation.field_total(hypttr)) or goal_nodes != float(
            Evaluation.field_total(goalttr)
        ):
            logger.warning("SIZE PROBLEM : totalNodes {} or goalNodes {}", total_nodes, goal_nodes)
            logger.warning("field total hypttr {}", float(Evaluation.field_total(hypttr)))
            logger.warning("field total goalttr{}", float(Evaluation.field_total(goalttr)))
            logger.warning("{}", hypttr)
            logger.warning("{}", goalttr)
        res = EvaluationResult.from_counts(total_nodes, goal_nodes, nodes_mapped)
        logger.debug("precision = {}", res.get_precision())
        logger.debug("recall = {}", res.get_recall())
        logger.debug("f-score = {}", res.get_f_score())
        return res

    def precision_recall_macro(
        self,
        pairs: Iterable[tuple[TTRRecordType, TTRRecordType]],
    ) -> list[float]:
        """Macro-average P/R/F1 after deheading manifest heads (Java ``precisionRecallMacro``)."""
        items = list(pairs)
        if not items:
            return [0.0, 0.0, 0.0]
        overall_p = overall_r = overall_f = 0.0
        empty = TTRRecordType.parse("[]")
        assert empty is not None
        for hyp, goal in items:
            hypttr = hyp if hyp is not None else empty
            goalttr = goal if goal is not None else empty
            try:
                head = TTRLabel("head")
                hyp_rec = hypttr.get_record()
                goal_rec = goalttr.get_record()
                if head in hyp_rec and hyp_rec[head].get_type() is not None:
                    hypttr = self.remove_label(hypttr, head)
                if head in goal_rec and goal_rec[head].get_type() is not None:
                    goalttr = self.remove_label(goalttr, head)
                pr = self.precision_recall(hypttr, goalttr)
                logger.debug("Eval scores for {} and {} are:", hypttr, goalttr)
                logger.debug("precision = {}", pr.get_precision())
                logger.debug("recall = {}", pr.get_recall())
                logger.debug("f-score = {}", pr.get_f_score())
                overall_p += pr.get_precision()
                overall_r += pr.get_recall()
                overall_f += pr.get_f_score()
            except Exception:  # noqa: BLE001
                logger.error("COULD not evaluate \n{}\n and \n{}", hypttr, goalttr)
        n = float(len(items))
        precision = overall_p / n
        recall = overall_r / n
        fscore = overall_f / n
        logger.info("OVERALL MACRO precision = {}", precision)
        logger.info("OVERALL MACRO recall = {}", recall)
        logger.info("OVERALL MACRO f-score = {}", fscore)
        return [precision, recall, fscore]

    def precision_recall_micro(
        self,
        pairs: Iterable[tuple[TTRRecordType, TTRRecordType]],
    ) -> EvaluationResult:
        """Micro-average P/R/F1 after deheading manifest heads (Java ``precisionRecallMicro``)."""
        overall_total = overall_goal = overall_mapped = 0.0
        for hyp, goal in pairs:
            hypttr = hyp
            goalttr = goal
            logger.info("checking {} verses {}", hypttr, goalttr)
            try:
                head = TTRLabel("head")
                hyp_rec = hypttr.get_record()
                goal_rec = goalttr.get_record()
                if head in hyp_rec and hyp_rec[head].get_type() is not None:
                    hypttr = self.remove_label(hypttr, head)
                if head in goal_rec and goal_rec[head].get_type() is not None:
                    goalttr = self.remove_label(goalttr, head)
                overall_total += self.total_nodes_mapped(hypttr, hypttr)
                overall_goal += self.total_nodes_mapped(goalttr, goalttr)
                overall_mapped += self.total_nodes_mapped(hypttr, goalttr)
            except Exception:  # noqa: BLE001
                logger.error("Could NOT do MICRO P and R on {} and {}", hypttr, goalttr)
        logger.info("OVERALL MICRO precision results: ")
        res = EvaluationResult.from_counts(overall_total, overall_goal, overall_mapped)
        logger.info("precision = {}", res.get_precision())
        logger.info("recall = {}", res.get_recall())
        logger.info("f-score = {}", res.get_f_score())
        return res

    def find_best_ttr_interpretation(
        self,
        predictions: Iterable[TTRRecordType],
        gold_rt: TTRRecordType,
    ) -> TTRRecordType | None:
        """Return mutual-subsumption match if any, else max-F1 prediction (Java ``findBestTTRInterpretation``)."""
        preds = list(predictions)
        logger.info("Finding best interpretation given the gold RT: {}", gold_rt)
        logger.info("Checking against {} predictions...", len(preds))
        best_eval: EvaluationResult | None = None
        best_pred: TTRRecordType | None = None
        for pred_rt in preds:
            logger.info("Checking prediction: {}", pred_rt)
            if pred_rt.subsumes(gold_rt) and gold_rt.subsumes(pred_rt):
                logger.info("Found perfect match: {}", pred_rt)
                return pred_rt
            eval_res = self.precision_recall(pred_rt, gold_rt)
            if best_eval is None or eval_res.get_f_score() > best_eval.get_f_score():
                best_eval = eval_res
                best_pred = pred_rt
                logger.info(
                    "New best prediction: {} with f-score: {}",
                    pred_rt,
                    eval_res.get_f_score(),
                )
            else:
                logger.info(
                    "Skipping prediction (we've had better!): {} with f-score: {}",
                    pred_rt,
                    eval_res.get_f_score(),
                )
        return best_pred


# Module-level singleton: instance methods stay on the class; static aliases call this.
_EVAL = Evaluation()

# Preserve unbound instance methods before installing static call-site aliases.
_INSTANCE_TOTAL_NODES = Evaluation.total_nodes_mapped
_INSTANCE_PRECISION_RECALL = Evaluation.precision_recall
_INSTANCE_PRECISION_RECALL_MACRO = Evaluation.precision_recall_macro
_INSTANCE_PRECISION_RECALL_MICRO = Evaluation.precision_recall_micro
_INSTANCE_FIND_BEST = Evaluation.find_best_ttr_interpretation


def _total_nodes_mapped(hyp: TTRRecordType | None, goal: TTRRecordType | None) -> float:
    """Static wrapper for instance ``total_nodes_mapped``."""
    return _INSTANCE_TOTAL_NODES(_EVAL, hyp, goal)


def _precision_recall(hyp: TTRRecordType, goal: TTRRecordType) -> EvaluationResult:
    """Static wrapper for instance ``precision_recall``."""
    return _INSTANCE_PRECISION_RECALL(_EVAL, hyp, goal)


def _precision_recall_macro(
    pairs: Iterable[tuple[TTRRecordType, TTRRecordType]],
) -> list[float]:
    """Static wrapper for instance ``precision_recall_macro``."""
    return _INSTANCE_PRECISION_RECALL_MACRO(_EVAL, pairs)


def _precision_recall_micro(
    pairs: Iterable[tuple[TTRRecordType, TTRRecordType]],
) -> EvaluationResult:
    """Static wrapper for instance ``precision_recall_micro``."""
    return _INSTANCE_PRECISION_RECALL_MICRO(_EVAL, pairs)


def _find_best_ttr_interpretation(
    predictions: Iterable[TTRRecordType],
    gold: TTRRecordType,
) -> TTRRecordType | None:
    """Static wrapper for instance ``find_best_ttr_interpretation`` (Java arg order)."""
    return _INSTANCE_FIND_BEST(_EVAL, predictions, gold)


Evaluation.total_nodes_mapped = staticmethod(_total_nodes_mapped)  # type: ignore[method-assign]
Evaluation.precision_recall = staticmethod(_precision_recall)  # type: ignore[method-assign]
Evaluation.precision_recall_macro = staticmethod(_precision_recall_macro)  # type: ignore[method-assign]
Evaluation.precision_recall_micro = staticmethod(_precision_recall_micro)  # type: ignore[method-assign]
Evaluation.find_best_ttr_interpretation = staticmethod(_find_best_ttr_interpretation)  # type: ignore[method-assign]

Evaluation.fieldTotal = staticmethod(Evaluation.field_total)  # type: ignore[attr-defined,method-assign]
Evaluation.totalNodesMapped = staticmethod(_total_nodes_mapped)  # type: ignore[attr-defined,method-assign]
Evaluation.precisionRecall = staticmethod(_precision_recall)  # type: ignore[attr-defined,method-assign]
Evaluation.precisionRecallMacro = staticmethod(_precision_recall_macro)  # type: ignore[attr-defined,method-assign]
Evaluation.precisionRecallMicro = staticmethod(_precision_recall_micro)  # type: ignore[attr-defined,method-assign]
Evaluation.findBestTTRInterpretation = staticmethod(_find_best_ttr_interpretation)  # type: ignore[attr-defined,method-assign]
Evaluation.removeLabel = Evaluation.remove_label  # type: ignore[attr-defined]
Evaluation.orderByManifest = Evaluation.order_by_manifest  # type: ignore[attr-defined]
Evaluation.maximalMapping = Evaluation.maximal_mapping  # type: ignore[attr-defined]
