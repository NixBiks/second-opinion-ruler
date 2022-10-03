# Second Opinion Ruler

`second_opinion_ruler` is a [spaCy](https://spacy.io/) component that extends [`SpanRuler`](https://spacy.io/usage/rule-based-matching#spanruler) with a second opinion. For _each_ pattern you can provide a callback (available in [`registry.misc`](https://spacy.io/api/top-level/#registry)) on the matched [`Span`](https://spacy.io/api/span/#_title) - with this you can decide to discard the match, add additional spans to the match and/or mutate the matched span, e.g. add a parsed `datetime` to a custom attribute.

## Installation

```
pip install second_opinion_ruler
```

## Usage

```python
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
ruler = nlp.add_pipe("second_opinion_ruler", config={
    "validate": True,
    "annotate_ents": True,
})

# add a pattern with a second opinion handler (on_match)
ruler.add_patterns([
    {
        "label": "DATE",
        "pattern": "21.04.1986",
        "on_match": {
            "id": "to_datetime.v1",
            "kwargs": {"format": "%d.%m.%Y", "attr": "my_date"},
        },
    }
])

doc = nlp("This date 21.04.1986 will be a DATE entity while the structured information will be extracted to `Span._.extructure`")

# verify
assert doc.ents[0]._.date == datetime.datetime(1986, 4, 21)
```
