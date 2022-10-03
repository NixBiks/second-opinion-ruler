import dataclasses
import re
import typing as t
import warnings

from spacy.errors import Errors
from spacy.language import Language
from spacy.pipeline.span_ruler import DEFAULT_SPANS_KEY, PatternType, SpanRuler
from spacy.tokens import Doc, Span
from spacy.util import registry
from typing_extensions import Unpack

DateFormat = str
SinglePattern = str | list[dict[str, t.Any]]


class _OnMatchArgs(t.TypedDict):
    id: str


class OnMatchArgs(_OnMatchArgs, total=False):
    args: list[t.Any]
    kwargs: dict[str, t.Any]


class _Pattern(t.TypedDict):
    label: str
    pattern: SinglePattern


class Pattern(_Pattern, total=False):
    id: str
    on_match: OnMatchArgs


class MatchLabelById(t.TypedDict):
    id: str
    label: str
    on_match: OnMatchArgs | None


@dataclasses.dataclass
class Rule:
    patterns: list[SinglePattern]
    extruct: str
    label: str


@dataclasses.dataclass
class CompiledRule(Rule):
    compiled_extruct: re.Pattern


@Language.factory(
    "second_opinion_ruler",
    assigns=["doc.spans"],
    default_config={
        "spans_key": DEFAULT_SPANS_KEY,
        "spans_filter": None,
        "annotate_ents": False,
        "ents_filter": {"@misc": "spacy.first_longest_spans_filter.v1"},
        "phrase_matcher_attr": None,
        "validate": False,
        "overwrite": True,
        "scorer": {
            "@scorers": "spacy.overlapping_labeled_spans_scorer.v1",
            "spans_key": DEFAULT_SPANS_KEY,
        },
    },
    default_score_weights={
        f"spans_{DEFAULT_SPANS_KEY}_f": 1.0,
        f"spans_{DEFAULT_SPANS_KEY}_p": 0.0,
        f"spans_{DEFAULT_SPANS_KEY}_r": 0.0,
        f"spans_{DEFAULT_SPANS_KEY}_per_type": None,
    },
)
def make_second_opinion_ruler(
    # **kwargs: Unpack[KWargs],
    nlp: Language,
    name: str,
    spans_key: t.Optional[str],
    spans_filter: t.Optional[
        t.Callable[[t.Iterable[Span], t.Iterable[Span]], t.Iterable[Span]]
    ],
    annotate_ents: bool,
    ents_filter: t.Callable[[t.Iterable[Span], t.Iterable[Span]], t.Iterable[Span]],
    phrase_matcher_attr: t.Optional[int | str],
    validate: bool,
    overwrite: bool,
    scorer: t.Optional[t.Callable],
):
    return SecondOpinionRuler(
        nlp=nlp,
        name=name,
        spans_key=spans_key,
        spans_filter=spans_filter,
        annotate_ents=annotate_ents,
        ents_filter=ents_filter,
        phrase_matcher_attr=phrase_matcher_attr,
        validate=validate,
        overwrite=overwrite,
        scorer=scorer,
    )


class SecondOpinionRuler(SpanRuler):
    def __init__(self, nlp: Language, name: str = "second_opinion_ruler", **kwargs):
        super().__init__(nlp, name, **kwargs)
        self._match_label_id_map: dict[int, MatchLabelById] = {}
        self._patterns: list[Pattern] = []

    def add_patterns(self, patterns: list[Pattern]) -> None:
        """Add patterns to the span ruler. A pattern can either be a token
        pattern (list of dicts) or a phrase pattern (string). For example:
        {'label': 'ORG', 'pattern': 'Apple'}
        {'label': 'ORG', 'pattern': 'Apple', 'id': 'apple'}
        {'label': 'GPE', 'pattern': [{'lower': 'san'}, {'lower': 'francisco'}]}

        patterns (list): The patterns to add.

        DOCS: https://spacy.io/api/spanruler#add_patterns
        """

        # disable the nlp components after this one in case they haven't been
        # initialized / deserialized yet
        try:
            current_index = -1
            for i, (name, pipe) in enumerate(self.nlp.pipeline):
                if self == pipe:
                    current_index = i
                    break
            subsequent_pipes = [pipe for pipe in self.nlp.pipe_names[current_index:]]
        except ValueError:
            subsequent_pipes = []
        with self.nlp.select_pipes(disable=subsequent_pipes):
            phrase_pattern_labels = []
            phrase_pattern_texts = []
            for entry in patterns:
                p_label = entry["label"]
                p_id = entry.get("id", "")
                p_on_match = entry.get("on_match")
                p_on_match_id = "" if p_on_match is None else p_on_match["id"]
                label = repr((p_label, p_id, p_on_match_id))
                self._match_label_id_map[self.nlp.vocab.strings.as_int(label)] = {
                    "label": p_label,
                    "id": p_id,
                    "on_match": p_on_match,
                }
                if isinstance(entry["pattern"], str):
                    phrase_pattern_labels.append(label)
                    phrase_pattern_texts.append(entry["pattern"])
                elif isinstance(entry["pattern"], list):
                    self.matcher.add(label, [entry["pattern"]])
                else:
                    raise ValueError(Errors.E097.format(pattern=entry["pattern"]))
                self._patterns.append(entry)
            for label, pattern in zip(
                phrase_pattern_labels,
                self.nlp.pipe(phrase_pattern_texts),
            ):
                self.phrase_matcher.add(label, [pattern])

    def _get_spans(self, span: Span, on_match: OnMatchArgs | None) -> list[Span]:
        if on_match is None:
            return [span]
        else:
            fn: None | t.Callable[[t.Any], list[Span]] = registry.misc.get(
                on_match["id"]
            )
            if fn is None:
                warnings.warn(Errors.W001.format(on_match["id"]))
                return [span]

            return fn(span, *on_match.get("args", []), **on_match.get("kwargs", {}))

    def match(self, doc: Doc) -> list[Span]:
        self._require_patterns()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="\\[W036")
            matches = t.cast(
                list[t.Tuple[int, int, int]],
                list(self.matcher(doc)) + list(self.phrase_matcher(doc)),
            )

        deduplicated_matches = set(
            span
            for m_id, start, end in matches
            for span in self._get_spans(
                Span(
                    doc,
                    start,
                    end,
                    label=self._match_label_id_map[m_id]["label"],
                    span_id=self._match_label_id_map[m_id]["id"],
                ),
                self._match_label_id_map[m_id]["on_match"],
            )
            if start != end
        )
        return sorted(list(deduplicated_matches))  # type: ignore
