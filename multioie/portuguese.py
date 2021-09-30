import logging
from pathlib import Path

from multioie.model.AllenNLP import AllenOpenIE, MultiOIEToken, LearningType, EmbeddingType, OptimizerType
from multioie.utils.contractions import transform_portuguese_contractions, clean_extraction
from allennlp.data.tokenizers import SpacyTokenizer

from multioie.utils.dataset import read_dataset


class PortugueseOIE:

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.tokenizer = SpacyTokenizer(language="pt_core_news_lg", split_on_spaces=True, parse=True)
        self.model = None

    def predict(self, tokens):
        if self.model is None:
            self.model = AllenOpenIE(model_folder=self.model_path)

        return self.model.predict(tokens)

    def predict_str(self, input_str: str):

        processed_str = clean_extraction(input_str)
        processed_str = transform_portuguese_contractions(processed_str)

        model = AllenOpenIE(model_folder=self.model_path)

        spacy_tokens = self.tokenizer.tokenize(processed_str)
        tokens = [MultiOIEToken(token=x.text, pos=x.pos_, dep=x.dep_) for x in spacy_tokens]

        prediction = model.predict(tokens)
        print(prediction)

    def train(
        self,
        input_path: Path,
        destination_model_path: Path,
        max_iterations: int = 50,
        layers: int = 1,
        network=LearningType.SRU,
        embedding=EmbeddingType.SELF_200,
        optimizer=OptimizerType.MADGRAD,
        hidden_dimension=512,
    ):
        model = AllenOpenIE(
            max_iterations=max_iterations,
            model_folder=destination_model_path,
            network=network,
            layers=layers,
            embedding=embedding,
            optimizer=optimizer,
            hidden_dimension=hidden_dimension,
            batch_size=32,
            patience=40,
            lr_patience=20
        )

        tokens, tags = read_dataset(input_path)

        model.train(tokens=tokens, tags=tags)

        logging.info("Model training complete")
