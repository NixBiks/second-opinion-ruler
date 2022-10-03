import datetime

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.util import registry

from second_opinion_ruler import SecondOpinionRuler


@pytest.fixture(scope="module")
def nlp():
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "second_opinion_ruler",
        config={
            "validate": True,
            "annotate_ents": True,
        },
    )
    return nlp


@pytest.fixture(scope="module")
def ruler(nlp: Language):
    return nlp.get_pipe("second_opinion_ruler")


# prepare
Span.set_extension("my_date", default=None)


@registry.misc("to_datetime.v1")
def to_datetime(span: Span, format: str, attr: str = "date") -> list[Span]:

    date = datetime.datetime.strptime(span.text, format)
    span._.set(attr, date)

    return [span]


@registry.misc("no_match_on_large_doc.v1")
def no_match_on_large_doc(span: Span, max_size=5) -> list[Span]:
    if len(span.doc) > max_size:
        return []

    return [span]


def test_simple(nlp: Language, ruler: SecondOpinionRuler):
    ruler.clear()
    ruler.add_patterns([{"label": "DATE", "pattern": "21.04.1986"}])

    doc = nlp("My birthday is 21.04.1986")
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == "DATE"
    assert doc.ents[0].text == "21.04.1986"


def test_args_match(nlp: Language, ruler: SecondOpinionRuler):

    # add patterns
    ruler.clear()
    ruler.add_patterns(
        [
            {
                "label": "DATE",
                "pattern": "21.04.1986",
                "on_match": {
                    "id": "to_datetime.v1",
                    "args": ["%d.%m.%Y", "my_date"],
                },
            }
        ]
    )

    doc = nlp("My birthday is 21.04.1986")
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == "DATE"
    assert doc.ents[0].text == "21.04.1986"
    assert doc.ents[0]._.my_date == datetime.datetime(1986, 4, 21)


def test_kwargs_match(nlp: Language, ruler: SecondOpinionRuler):

    # add patterns
    ruler.clear()
    ruler.add_patterns(
        [
            {
                "label": "DATE",
                "pattern": "21.04.1986",
                "on_match": {
                    "id": "to_datetime.v1",
                    "kwargs": {"format": "%d.%m.%Y", "attr": "my_date"},
                },
            }
        ]
    )

    doc = nlp("My birthday is 21.04.1986")
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == "DATE"
    assert doc.ents[0].text == "21.04.1986"
    assert doc.ents[0]._.my_date == datetime.datetime(1986, 4, 21)


def test_ignore_match_on_large_doc(nlp: Language, ruler: SecondOpinionRuler):

    # add patterns
    ruler.clear()
    ruler.add_patterns(
        [
            {
                "label": "LOREM",
                "pattern": "lorem",
                "on_match": {
                    "id": "no_match_on_large_doc.v1",
                },
            }
        ]
    )

    doc = nlp("lorem ipsum dolor sit amet")
    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == "LOREM"

    doc = nlp("lorem ipsum dolor sit amet lorem ipsum dolor sit amet")
    assert len(doc.ents) == 0


def test_example():
    import spacy
    from spacy.tokens import Span
    from spacy.util import registry

    # create date as custom attribute extension
    Span.set_extension("date", default=None)

    # add datetime parser to registry.misc
    # IMPORTANT: first argument has to be Span and the return type has to be list[Span]
    @registry.misc("to_datetime.v1")
    def to_datetime(span: Span, format: str, attr: str = "date") -> list[Span]:

        # parse the date
        date = datetime.datetime.strptime(span.text, format)

        # add the parsed date to the custom attribute
        span._.set(attr, date)

        # just return matched span
        return [span]

    # load a model
    nlp = spacy.blank("en")

    # add the second opinion ruler
    ruler = nlp.add_pipe(
        "second_opinion_ruler",
        config={
            "validate": True,
            "annotate_ents": True,
        },
    )

    # add a pattern with a second opinion handler (on_match)
    ruler.add_patterns(  # type: ignore
        [
            {
                "label": "DATE",
                "pattern": "21.04.1986",
                "on_match": {
                    "id": "to_datetime.v1",
                    "kwargs": {"format": "%d.%m.%Y", "attr": "date"},
                },
            }
        ]
    )

    doc = nlp(
        "This date 21.04.1986 will be a DATE entity while the structured information will be extracted to `Span._.extructure`"
    )

    # verify
    assert doc.ents[0]._.date == datetime.datetime(1986, 4, 21)
