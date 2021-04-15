import logging
from pathlib import Path
from typing import List

from allennlp.data.tokenizers import SpacyTokenizer

from multioie.model.AllenNLP import AllenOpenIE, MultiOIEToken
import typer

app = typer.Typer()


class MultiOIE:
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def predict(self, file_path: Path):

        pass

    def _read_dataset(self, path: Path):

        tokens: List[List[MultiOIEToken]] = []
        tags: List[List[str]] = []

        # Our tokenizer
        tokenizer = SpacyTokenizer(language="pt_core_news_lg", split_on_spaces=True)

        with path.open("r", encoding="utf-8") as f:

            token_buffer = []
            tags_buffer = []
            for line in f:
                line = line.strip()
                if len(line) < 2:

                    if len(token_buffer) > 0:
                        str_phrase = " ".join(token_buffer)
                        spacy_tokens = tokenizer.tokenize(str_phrase)
                        tokens.append(
                            [MultiOIEToken(token=x.text, pos=x.pos_) for x in spacy_tokens]
                        )
                        tags.append(tags_buffer)

                    token_buffer = []
                    tags_buffer = []
                    continue

                parts = line.split("\t")
                # TODO parse the other parts of line
                word = parts[3].strip()
                tag = parts[11].strip()

                token_buffer.append(word)
                tags_buffer.append(tag)

        # Last item
        if len(token_buffer) > 0:
            str_phrase = " ".join(token_buffer)
            spacy_tokens = tokenizer.tokenize(str_phrase)
            tokens.append([MultiOIEToken(token=x.text, pos=x.pos_) for x in spacy_tokens])
            tags.append(tags_buffer)

        return tokens, tags

    def train(self, input_path: Path, destination_model_path: Path):
        model = AllenOpenIE(max_iterations=self.max_iterations, model_folder=destination_model_path)

        tokens, tags = self._read_dataset(input_path)

        model.train(tokens=tokens, tags=tags, validation_tokens=tokens, validation_tags=tags)

        logging.info("Model training complete")


@app.command()
def train(input_path_str: str, model_output_path_str: Path, max_iterations: int = 50):
    input_path = Path(input_path_str).resolve()
    output_path = Path(model_output_path_str).resolve()

    oie_system = MultiOIE(max_iterations=max_iterations)
    oie_system.train(input_path=input_path, destination_model_path=output_path)


@app.command()
def predict(input_str: str, model_path: Path):
    pass


if __name__ == "__main__":
    app()
