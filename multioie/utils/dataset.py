from itertools import groupby
from pathlib import Path
from typing import List

from allennlp.data.tokenizers import SpacyTokenizer
from allennlp_models.common.ontonotes import Ontonotes

from multioie.model.AllenNLP import MultiOIEToken


def read_dataset(path: Path):

    tokens: List[List[MultiOIEToken]] = []
    tags: List[List[str]] = []

    # Our tokenizer
    tokenizer = SpacyTokenizer(language="pt_core_news_lg", split_on_spaces=True, parse=True)

    _reader = Ontonotes()

    files = list(Path(path).glob("*.conll"))

    for file in files:
        print(f"Processing {file=}")

        grouped_sentences = [
            list(g)
            for k, g in groupby(_reader.sentence_iterator(file), lambda x: x.sentence_id)
        ]

        for sentences in grouped_sentences:
            str_phrase = " ".join(sentences[0].words)
            spacy_tokens = tokenizer.tokenize(str_phrase)

            extractions = []
            for sentence in sentences:
                if sentence.srl_frames:
                    for (_, sent_tags) in sentence.srl_frames:
                        extractions.append(sent_tags)

            if len(extractions) == 0:
                # Sentence contains no predicates.
                extractions.append(["O" for _ in spacy_tokens])

            # Now add
            tokens.append(
                [MultiOIEToken(token=x.text, pos=x.pos_, dep=x.dep_) for x in spacy_tokens]
            )
            tags.append(extractions)

    return tokens, tags