import gc
import itertools
import logging
import pickle
import re
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import random
from typing import Iterator, List, Iterable, Any, Dict, Union

import torch
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.training import Checkpointer, TrainerCallback
from allennlp_models.structured_prediction import SemanticRoleLabeler, OpenIePredictor
from flair.embeddings import FlairEmbeddings
from flair.file_utils import cached_path
from torch.nn import LSTM

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SequenceLabelField, TextField, MetadataField
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer, PretrainedTransformerIndexer,
)
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary

from allennlp_models.tagging.models import CrfTagger
from allennlp.nn import RegularizerApplicator
from allennlp.nn.regularizers import L1Regularizer

from allennlp.common import Params
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import ReduceOnPlateauLearningRateScheduler
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder, ElmoTokenEmbedder, \
    PretrainedTransformerEmbedder
from allennlp.predictors import Predictor
from torch.optim import SGD
from torch_optimizer import RAdam

from multioie.model.allen_helpers.array_indexer import ArrayIndexer
from multioie.model.allen_helpers.broka_text_field_embedder import BrokaTextFieldEmbedder
from multioie.model.allen_helpers.broka_trainer import BrokaTrainer
from multioie.model.allen_helpers.flair_char_indexer import FlairCharIndexer
from multioie.model.allen_helpers.flair_token_embedder import FlairEmbedder
from multioie.model.allen_helpers.my_fast_transformers import MyMemTransformer
from multioie.model.featurizer import Featurizer
from multioie.model.openie_predictor import MultiOIEOpenIePredictor
from multioie.optimizers.madgrad_wd import Madgrad_wd
from multioie.utils.align import StringAligner

from multioie.utils.token_iter import gen_split_overlap

torch.manual_seed(1)

FLAIR_URL_BASE = "http://minio.potelo.com.br/asserts/flair_diario/"
_MAX_SIZE_TO_PROCESS = 1024 * 32  # É o limite para uma GPU de 8GB com o Flair 1024 BI
_MAX_OVERLAP_TO_PROCESS = 1024 * 3


@dataclass
class MultiOIEToken:
    token: str
    pos: str
    dep: str


class EmbeddingType(Enum):
    ELMO = 1
    GLOVE = 2
    FLAIR_DIARIOS_1024 = 3
    FLAIR_DIARIOS_BI_1024 = 4
    FLAIR_DIARIOS_2048 = 5
    FLAIR_DIARIOS_BI_2048 = 6
    SELF_100 = 7
    SELF_300 = 8
    SELF_200 = 9
    BERT_PT = 10


class OptimizerType(Enum):
    SGD = 1
    RADAM = 2
    MADGRAD = 3


class LearningType(Enum):
    LSTM = 1
    SRU = 2
    XTRANSFORMER = 13
    SRUPP = 14


class MyTrainerCallback(TrainerCallback):
    def __init__(self, callback, total_iterations):
        self.callback = callback
        self.total_iterations = total_iterations
        super(MyTrainerCallback, self).__init__(None)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        self.callback(epoch + 1, self.total_iterations, metrics)

    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        self.callback(epoch + 1, self.total_iterations, metrics)


@dataclass(init=False)
class TokenBroka(Token):
    __slots__ = ["word_shape_degen", "one_hot", "dep_tags"]
    word_shape_degen: str
    one_hot: bool
    dep_tags: str


class BrokaReader(DatasetReader):
    ONE_HOT_FEATURES = [
        "dummy_one_hot",
        # TODO - Use a real feature for ONE hot feature
    ]

    def __init__(
        self,
        limit=None,
        simple_text=False,
        one_hot_layout=None,
        use_elmo=True,
        use_flair=True,
        use_bert=False,
        use_flair_backward=False,
        flair_fw_embedding_path=None,
        flair_bw_embedding_path=None,
        use_char_cnn=True,
        replace_digits=True,
    ) -> None:
        super().__init__()
        self.limit = limit
        self.simple_text = simple_text
        self.one_hot_layout = one_hot_layout
        self.use_elmo = use_elmo
        self.use_flair = use_flair
        self.use_bert=use_bert
        self.use_flair_backward = use_flair_backward
        self.flair_fw_embedding_path = flair_fw_embedding_path
        self.flair_bw_embedding_path = flair_bw_embedding_path
        self.use_char_cnn = use_char_cnn
        self.replace_digits = replace_digits

    def token_to_one_hot_vector(self, token):
        features = []
        for name, value in self.one_hot_layout.items():
            item_value = str(token[name])
            if item_value in value:
                item_as_id = value[item_value]
            else:
                # print(f"{item_value} não existe em {value} ")
                item_as_id = value[next(iter(value))]  # Vamos pegar o primeiro, já que não tem
            num_labels = len(value)

            if num_labels < 3:  # Se so tem duas opcoes vamos colocar binario
                tensor = [item_as_id]
            else:
                tensor = [0] * num_labels
                tensor[item_as_id] = 1

            features.extend(tensor)

        return features

    def token_to_instances_to_prediction(self, tokens: List[object]) -> List[Instance]:
        # Find all verbs in the input sentence

        spacy_verb_indicator = [1 if token.pos in ["VERB", "AUX"] else 0 for token in tokens]

        verb_instances = self.get_verb_instances(spacy_verb_indicator)
        if len(verb_instances) == 0:
            verb_instances.append(spacy_verb_indicator)

        instances = []
        for verb_labels in verb_instances:

            sentence_field = self._create_sentence_field(tokens)

            fields = {
                "tokens": sentence_field,
                "verb_indicator": SequenceLabelField(verb_labels, sequence_field=sentence_field),
            }

            # Set metadata- Needed for SemanticRoleLabeler
            metadata_dict: Dict[str, Any] = {}

            if all(x == 0 for x in verb_labels):
                verb = None
                verb_index = None
            else:
                verb_index = verb_labels.index(1)
                verb = tokens[verb_index].token

            metadata_dict["words"] = [x.token for x in tokens]
            metadata_dict["verb"] = verb
            metadata_dict["verb_index"] = verb_index

            fields["metadata"] = MetadataField(metadata_dict)

            instances.append(Instance(fields))
        return instances

    def match_tags_to_verb_indicator(self, tags, verb_instances):
        results = []

        empty_tag = ["O" for _ in range(len(verb_instances[0]))]
        set_existing_instances = verb_instances.copy()
        # TODO - A Dynamic match would be better...
        for tag in tags:
            binary_tags = [1 if "-V" in x else 0 for x in tag]
            best_score = {"score": -1, "item": None, "pos": None}
            for pos, verb_instance in enumerate(verb_instances):
                score = sum(x == y and x == 1 for x, y in zip(binary_tags, verb_instance))

                if score > best_score["score"]:
                    best_score["score"] = score
                    best_score["item"] = verb_instance
                    best_score["pos"] = pos

            results.append((tag, best_score["item"]))
            if best_score["item"] and (best_score["item"] in set_existing_instances):
                set_existing_instances.remove(best_score["item"])

        for non_matching_instance in set_existing_instances:
            results.append((empty_tag, non_matching_instance))

        return results

    def token_to_instance(self, tokens: List[object], tags: List[str] = None) -> Instance:

        spacy_verb_indicator = [1 if token.pos in ["VERB", "AUX"] else 0 for token in tokens]

        verb_instances = self.get_verb_instances(spacy_verb_indicator)
        if len(verb_instances) == 0:
            verb_instances.append(spacy_verb_indicator)

        # Instance
        instances = []

        itens = self.match_tags_to_verb_indicator(tags=tags, verb_instances=verb_instances)

        for item_tags, verb_indicator in itens:

            sentence_field = self._create_sentence_field(tokens)

            fields = {
                "tokens": sentence_field,
                "verb_indicator": SequenceLabelField(verb_indicator, sequence_field=sentence_field),
            }

            metadata_dict: Dict[str, Any] = {}

            if all(x == 0 for x in verb_indicator):
                verb = None
                verb_index = None
            else:
                verb_index = verb_indicator.index(1)
                verb = tokens[verb_index].token

            metadata_dict["words"] = [x.token for x in tokens]
            metadata_dict["verb"] = verb
            metadata_dict["verb_index"] = verb_index

            if item_tags:
                if len(item_tags) != len(sentence_field):
                    raise Exception("Invalid number")
                label_field = SequenceLabelField(labels=item_tags, sequence_field=sentence_field)
                fields["tags"] = label_field

                metadata_dict["gold_tags"] = item_tags

            fields["metadata"] = MetadataField(metadata_dict)

            instances.append(Instance(fields))

        return instances

    def get_verb_instances(self, verb_indicator):
        buffer_results = []
        verb_instances = []
        for verb_slice in [list(g) for k, g in itertools.groupby(verb_indicator)]:
            if any(x == 1 for x in verb_slice):
                new_instance = [0 for _ in range(len(buffer_results))]
                new_instance.extend(verb_slice)
                missing_zeros = len(verb_indicator) - len(new_instance)
                if missing_zeros > 0:
                    new_instance.extend([0 for _ in range(missing_zeros)])

                verb_instances.append(new_instance)

            buffer_results.extend(verb_slice)
        return verb_instances

    def _create_sentence_field(self, tokens):
        token_list = []
        featurizer = Featurizer()
        count = 0
        for token in tokens:
            count += 1

            if self.simple_text:
                transformed_token = token
            else:
                transformed_token = featurizer.agregado(token)

            # Ignored 'word_shape', 'no_accents'

            if not self.simple_text:
                if self.replace_digits:
                    token = re.sub(r"\d", "0", transformed_token["token"])
                else:
                    token = transformed_token["token"]
                token = TokenBroka(token)  # type: ignore
                token.word_shape_degen = transformed_token["word_shape_degen"]  # type: ignore
                token.dep_tags = transformed_token["dep"]
                token.pos_ = transformed_token["pos"]
                # token.one_hot = self.token_to_one_hot_vector(transformed_token)  # type: ignore
            else:
                if self.replace_digits:
                    token = Token(re.sub(r"\d", "0", transformed_token))
                else:
                    token = Token(transformed_token)

            token_list.append(token)

        token_indexer = {}
        if not self.use_bert:
            token_indexer["tokens"]: SingleIdTokenIndexer()

        if self.use_char_cnn:
            token_indexer["token_characters"] = TokenCharactersIndexer(min_padding_length=3)
        if not self.simple_text:
            token_indexer["word_shape_degen"] = SingleIdTokenIndexer(
                namespace="word_shape_degen", feature_name="word_shape_degen"
            )
            token_indexer["dep_tags"] = SingleIdTokenIndexer(
                namespace="dep_tags", feature_name="dep_tags"
            )
            # token_indexer["one_hot"] = ArrayIndexer(namespace="one_hot", feature_name="one_hot")
        if self.use_elmo:
            token_indexer["elmo"] = ELMoTokenCharactersIndexer()
        if self.use_flair and self.flair_fw_embedding_path is not None:
            token_indexer["flair"] = FlairCharIndexer(self.flair_fw_embedding_path)
        if self.flair_bw_embedding_path is not None:
            token_indexer["flair-back"] = FlairCharIndexer(self.flair_bw_embedding_path)
        if self.use_bert:
            token_indexer["bert"] = PretrainedTransformerIndexer(model_name="neuralmind/bert-large-portuguese-cased")

        return TextField(token_list, token_indexers=token_indexer)

    def read(self, dataset) -> Iterator[Instance]:
        tokens, tags = dataset
        for doc_tokens, doc_tags in zip(tokens, tags):
            for instance in self.token_to_instance(doc_tokens, doc_tags):
                yield instance

    def count_one_hot(self, documents):
        if self.simple_text:
            return None

        counter_set = dict()
        for feat in self.ONE_HOT_FEATURES:
            counter_set[feat] = set()

        for doc in documents:
            featurizer = Featurizer()

            for token in doc:
                transformed_token = featurizer.agregado(token)

                for feat in self.ONE_HOT_FEATURES:
                    counter_set[feat].add(transformed_token[feat])

        counter = dict()
        for name, values in counter_set.items():
            if len(values) > 1:
                counter[name] = dict()
                for pos, value in enumerate(values):
                    counter[name][str(value)] = pos

        self.one_hot_layout = counter
        return counter


class MaxSizeFillBucketBatchSampler(BucketBatchSampler):
    def __init__(self, *args, **kwargs):
        self.max_size = kwargs["max_size"]
        del kwargs["max_size"]
        super().__init__(*args, **kwargs)

    def __iter__(self) -> Iterable[List[int]]:
        indices, sizes = self._argsort_by_padding(self.data_source)
        batches = []
        actual_size = 0
        actual_batch: List[int] = []
        for indice, size in zip(indices, sizes):
            size = size[0]
            if (len(actual_batch) > 0 and ((actual_size + size) >= self.max_size)) or (
                len(actual_batch) > self.batch_size
            ):
                batches.append(actual_batch)
                actual_batch = []
                actual_size = 0

            actual_size += size
            actual_batch.append(indice)

        # last one
        if len(actual_batch) > 0:
            batches.append(actual_batch)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        actual_count = 0
        internal_batch_count = 0
        internal_batch_size = 0
        for i in self.data_source:
            actual_size = len(next(iter(i.fields.values())))

            if ((internal_batch_count + actual_size) >= self.max_size) or (
                internal_batch_size > self.batch_size
            ):
                actual_count += 1
                internal_batch_count = 0
                internal_batch_size = 0
            internal_batch_count += actual_size
            internal_batch_size += 1

        if internal_batch_size > 0:
            actual_count += 1
        return actual_count


class AllenOpenIE:
    def __init__(
        self,
        model_folder: Path = None,
        max_iterations=150,
        network: LearningType = LearningType.LSTM,
        hidden_dimension=512,
        layers=2,
        dropout_layers=0.5,
        dropout_crf=0.2,
        c1: float = 0.0,
        c2: float = 0.0,
        learning_rate=0.005,
        bidirectional=True,
        token_limit=None,
        batch_size=8,
        simple_text=False,
        embedding: EmbeddingType = EmbeddingType.SELF_200,
        cache_folder=None,
        use_bucket_sampler=True,
        shuffle=True,
        use_char_cnn=False,
        replace_digits=True,
        optimizer=OptimizerType.MADGRAD,
        max_tokens_per_batch=_MAX_SIZE_TO_PROCESS,
        predict_window=_MAX_SIZE_TO_PROCESS,
        predict_overlap=_MAX_OVERLAP_TO_PROCESS,
        train_window=_MAX_SIZE_TO_PROCESS,
        train_overlap=_MAX_OVERLAP_TO_PROCESS,
        predict_batch_size=8,
        lr_patience=10,
        patience=30,
        # Transformers Exclusive
        positional_encoding="axial",
        num_attention_heads=4,
        query_dimensions=768,
    ):
        self.model_folder = model_folder
        self.max_iterations = max_iterations
        self.network = network
        self.layers = layers
        self.hidden_dimension = hidden_dimension
        self.c1 = c1
        self.c2 = c2
        self.bidirectional = bidirectional
        self.dropout_layers = dropout_layers
        self.dropout_crf = dropout_crf
        self.learning_rate = learning_rate
        self.token_limit = token_limit
        self.batch_size = batch_size
        self.simple_text = simple_text
        self.one_hot_layout = None
        self.embedding: EmbeddingType = embedding
        self.cache_folder = cache_folder
        self.use_bucket_sampler = use_bucket_sampler
        self.shuffle = shuffle
        self.train_metrics = None
        self.optimizer = optimizer
        self.use_char_cnn = use_char_cnn
        self.replace_digits = replace_digits

        self.max_tokens_per_batch = max_tokens_per_batch
        self.predict_window = predict_window
        self.predict_overlap = predict_overlap
        self.train_window = train_window
        self.train_overlap = train_overlap
        self.predict_batch_size = predict_batch_size
        self.lr_patience = lr_patience
        self.patience = patience

        # Transformers Exclusive
        self.positional_encoding = positional_encoding

        self.num_attention_heads = num_attention_heads
        self.query_dimensions = query_dimensions

        # Desativar Loggers
        logging.getLogger("allennlp.common.params").disabled = True
        logging.getLogger("allennlp.nn.initializers").disabled = True
        logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)
        logging.getLogger("urllib3.connectionpool").disabled = True

        # Setar os parameteros
        self.use_glove = False
        self.use_bert = False
        self.use_elmo = False
        self.use_flair = False
        self.use_flair_backward = False
        self.flair_fw_embedding_path = None
        self.flair_bw_embedding_path = None
        self.own_embedding_size = 100
        self.set_parameters_from_embedding()

        # Criar se nao existir
        if model_folder:
            self.model_folder.mkdir(parents=True, exist_ok=True)
            if cache_folder is None:
                self.cache_folder = self.model_folder

        self.model = None
        self.vocab = None

    def set_parameters_from_embedding(self):
        self.use_glove = False
        self.use_elmo = False
        self.use_flair = False
        self.use_flair_backward = False
        self.flair_fw_embedding_path = None
        self.flair_bw_embedding_path = None
        self.own_embedding_size = 100

        if self.embedding == EmbeddingType.GLOVE:
            self.use_glove = True
        elif self.embedding == EmbeddingType.ELMO:
            self.use_elmo = True
        elif self.embedding == EmbeddingType.FLAIR_DIARIOS_1024:
            self.use_flair = True
            self.flair_fw_embedding_path = cached_path(
                f"{FLAIR_URL_BASE}forward-1024.pt", cache_dir=Path("embeddings")
            )
        elif self.embedding == EmbeddingType.FLAIR_DIARIOS_2048:
            self.use_flair = True
            self.flair_fw_embedding_path = cached_path(
                f"{FLAIR_URL_BASE}forward-2048.pt", cache_dir=Path("embeddings")
            )
        elif self.embedding == EmbeddingType.FLAIR_DIARIOS_BI_1024:
            self.use_flair = True
            self.use_flair_backward = True
            self.flair_fw_embedding_path = cached_path(
                f"{FLAIR_URL_BASE}forward-1024.pt", cache_dir=Path("embeddings")
            )
            self.flair_bw_embedding_path = cached_path(
                f"{FLAIR_URL_BASE}backward-1024.pt", cache_dir=Path("embeddings")
            )
        elif self.embedding == EmbeddingType.FLAIR_DIARIOS_BI_2048:
            self.use_flair = True
            self.use_flair_backward = True
            self.flair_fw_embedding_path = cached_path(
                f"{FLAIR_URL_BASE}forward-2048.pt", cache_dir=Path("embeddings")
            )
            self.flair_bw_embedding_path = cached_path(
                f"{FLAIR_URL_BASE}backward-2048.pt", cache_dir=Path("embeddings")
            )
        elif self.embedding == EmbeddingType.SELF_100:
            self.own_embedding_size = 100
        elif self.embedding == EmbeddingType.SELF_200:
            self.own_embedding_size = 200
        elif self.embedding == EmbeddingType.SELF_300:
            self.own_embedding_size = 300
        elif self.embedding == EmbeddingType.BERT_PT:
            self.use_bert = True
        else:
            raise AttributeError

    def fit(self, x, y):
        return self.train(x, y)

    def get_model(self, vocab: Vocabulary, use_cache: bool = False) -> SemanticRoleLabeler:

        total_embedding_dim = 0
        # Vamos pegar o tamanho de features one-hot
        size_one_hot = 0
        if self.one_hot_layout:
            for name, values in self.one_hot_layout.items():
                tamanho = len(values)
                if tamanho < 3:
                    tamanho = 1
                size_one_hot += tamanho

        binary_feature_dim = 100
        total_embedding_dim += binary_feature_dim

        # TODO add one_hot
        # total_embedding_dim += size_one_hot

        # Now we need to construct the model.
        # We'll choose a size for our embedding layer and for the hidden layer of our LSTM.

        if self.use_bert:
            token_embedding = PretrainedTransformerEmbedder.from_params(
                Params({"model_name": "neuralmind/bert-large-portuguese-cased"})

            )
            total_embedding_dim += token_embedding.output_dim
        elif self.use_glove:
            # Usar o http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc

            token_embedding = Embedding.from_params(
                vocab=vocab,
                params=Params(
                    {
                        "pretrained_file": "http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip",
                        "embedding_dim": 300,
                    }
                ),
            )
            total_embedding_dim += 300

        else:
            token_embedding = Embedding(
                num_embeddings=vocab.get_vocab_size("tokens"),
                embedding_dim=self.own_embedding_size,
                trainable=True,
            )
            total_embedding_dim += self.own_embedding_size

        if self.use_bert:
            active_embedders = {
                "bert": token_embedding,
            }
        else:
            active_embedders = {
                "tokens": token_embedding,
            }

        if self.use_char_cnn:
            params = Params(
                {
                    "embedding": {"embedding_dim": 16, "vocab_namespace": "token_characters"},
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 16,
                        "num_filters": 128,
                        "ngram_filter_sizes": [3],
                        "conv_layer_activation": "relu",
                    },
                }
            )
            char_embedding = TokenCharactersEncoder.from_params(vocab=vocab, params=params)
            total_embedding_dim += 128
            active_embedders["token_characters"] = char_embedding

        # if self.use_elmo:
        #    elmo_emmbedder = ElmoTokenEmbedder.from_params(vocab=vocab, params=elmo_params)

        if self.use_flair:

            flair_model = FlairEmbeddings(self.flair_fw_embedding_path)

            if self.use_flair_backward:
                flair_model_back = FlairEmbeddings(self.flair_bw_embedding_path)

            cache_folder_flair_fw = None
            cache_folder_flair_bw = None
            if use_cache:
                if self.cache_folder is None:
                    logging.warning(
                        "Caching Flair Embeddings at a temporary folder, because model_folder is NOT set"
                    )
                else:
                    logging.warning(f"Caching Flair Embeddings at {self.cache_folder}")
                    cache_folder_flair_fw = self.cache_folder / "flair_forward"
                    cache_folder_flair_bw = self.cache_folder / "flair_backward"

            flair_emmbedder = FlairEmbedder(
                flair_model, caching=use_cache, caching_folder=cache_folder_flair_fw
            )
            total_embedding_dim += flair_emmbedder.output_dim
            active_embedders["flair"] = flair_emmbedder

            if self.use_flair_backward:
                flair_emmbedder_back = FlairEmbedder(
                    flair_model_back, caching=use_cache, caching_folder=cache_folder_flair_bw
                )
                total_embedding_dim += flair_emmbedder_back.output_dim
                active_embedders["flair-back"] = flair_emmbedder_back

        if not self.simple_text:
            shape_embedding = Embedding(
                num_embeddings=vocab.get_vocab_size("word_shape_degen"),
                embedding_dim=36,
                trainable=True,
            )
            total_embedding_dim += 36
            active_embedders["word_shape_degen"] = shape_embedding

            dep_embedding = Embedding(
                num_embeddings=vocab.get_vocab_size("dep_tags"),
                embedding_dim=36,
                trainable=True,
            )
            total_embedding_dim += 36
            active_embedders["dep_tags"] = dep_embedding

            # Fix to fit transformers head limitation
            # one_hot_embedding_size = 48
            # one_hot_embedding_size += -(total_embedding_dim + 36) % self.num_attention_heads
            #
            # # TODO also fix in simple text
            # one_hot_embedding = FeedForward(
            #     size_one_hot,
            #     num_layers=1,
            #     hidden_dims=one_hot_embedding_size,
            #     activations=torch.nn.ReLU(),
            #     dropout=0.0,
            # )
            #
            # active_embedders["one_hot"] = one_hot_embedding
            # total_embedding_dim += one_hot_embedding_size

        word_embeddings = BrokaTextFieldEmbedder(
            active_embedders, one_hot_size=size_one_hot, use_separated_one_hot=True
        )

        if self.network == LearningType.LSTM:
            network = LSTM(
                total_embedding_dim,
                self.hidden_dimension,
                num_layers=self.layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=0.0 if self.layers == 1 else self.dropout_layers,
            )
            encoder = PytorchSeq2SeqWrapper(network)
        elif self.network in [LearningType.SRU, LearningType.SRUPP]:
            from sru import SRU, SRUpp  # Python will cache multiple imports anyway
            from multioie.model.allen_helpers.sru_wrapper import MySRUWrapper

            if self.network == LearningType.SRU:
                network = SRU(
                    total_embedding_dim,
                    self.hidden_dimension,
                    num_layers=self.layers,  # number of stacking RNN layers
                    dropout=0.0 if self.layers == 1 else self.dropout_layers,
                    # dropout applied between RNN layers
                    bidirectional=self.bidirectional,  # bidirectional RNN
                    layer_norm=False,  # apply layer normalization on the output of each layer
                    normalize_after=False,
                    highway_bias=-2,  # initial bias of highway gate (<= 0)
                    rescale=False,  # whether to use scaling correction
                    nn_rnn_compatible_return=False,
                )
                encoder = MySRUWrapper(network, srupp=False)
            elif self.network == LearningType.SRUPP:
                network = SRUpp(
                    total_embedding_dim,
                    self.hidden_dimension,
                    proj_size=self.hidden_dimension // 2,
                    num_layers=self.layers,  # number of stacking RNN layers
                    attn_dropout=self.dropout_layers,
                    dropout=self.dropout_layers,
                    # dropout applied between RNN layers
                    bidirectional=self.bidirectional,  # bidirectional RNN
                    layer_norm=False,  # apply layer normalization on the output of each layer
                    normalize_after=False,
                    highway_bias=-2,  # initial bias of highway gate (<= 0)
                    rescale=False,  # whether to use scaling correction
                    nn_rnn_compatible_return=False,
                )
                encoder = MySRUWrapper(network, srupp=True)

        elif self.network in [LearningType.XTRANSFORMER]:

            encoder = MyMemTransformer(
                total_embedding_dim,
                num_layers=self.layers,
                num_attention_heads=self.num_attention_heads,
                feedforward_hidden_dim=self.hidden_dimension,
                network_type=self.network,
                dropout_prob=0.0 if self.layers == 1 else self.dropout_layers,
            )
        else:
            # LSTM + Attention
            # Compressive transformer
            raise AttributeError("Unknown network")

        regularization = None
        if self.c1 > 0.0:
            regularization = RegularizerApplicator([("", L1Regularizer(self.c1))])

        # Finally, we can instantiate the model.
        model = SemanticRoleLabeler(
            vocab=vocab,
            text_field_embedder=word_embeddings,
            binary_feature_dim=binary_feature_dim,
            encoder=encoder,
            regularizer=regularization,
            ignore_span_metric=True,
        )
        return model

    def train(
        self,
        tokens,
        tags,
        validation_tokens=None,
        validation_tags=None,
        resume=False,
        progress_callback=None,
    ):

        reader = BrokaReader(
            limit=self.token_limit,
            simple_text=self.simple_text,
            use_elmo=self.use_elmo,
            use_flair=self.use_flair,
            use_bert=self.use_bert,
            use_flair_backward=self.use_flair_backward,
            flair_fw_embedding_path=self.flair_fw_embedding_path,
            flair_bw_embedding_path=self.flair_bw_embedding_path,
            use_char_cnn=self.use_char_cnn,
            replace_digits=self.replace_digits,
        )

        one_hot_layout = reader.count_one_hot(tokens)

        train_dataset = reader.read((tokens, tags))

        if validation_tokens:
            validation_dataset = reader.read((validation_tokens, validation_tags))

        # Once we've read in the datasets, we use them to create our <code>Vocabulary</code>
        # (that is, the mapping[s] from tokens / labels to ids).
        if validation_tokens:
            vocab = Vocabulary.from_instances(itertools.chain(train_dataset, validation_dataset))
        else:
            vocab = Vocabulary.from_instances(train_dataset)

        if self.model_folder:
            check_point_folder = self.model_folder / "checkpoints"
        else:
            check_point_folder = Path(tempfile.TemporaryDirectory().name) / "checkpoints"

        if not check_point_folder.exists():
            check_point_folder.mkdir(parents=True, exist_ok=True)
        elif not resume:
            shutil.rmtree(check_point_folder, ignore_errors=True)
            check_point_folder.mkdir(parents=True, exist_ok=True)

        checkpointer = Checkpointer(check_point_folder, keep_most_recent_by_count=3)

        # Set variables
        self.one_hot_layout = one_hot_layout
        self.vocab = vocab

        self.model = self.get_model(self.vocab, use_cache=True)

        print(f"Started training a model with {self.get_config_state()}")

        gc.collect()

        if torch.cuda.is_available():
            cuda_device = 0
            self.model = self.model.cuda(cuda_device)
        else:
            cuda_device = -1

        if self.optimizer == OptimizerType.SGD:
            optimizer = SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.c2)
        elif self.optimizer == OptimizerType.RADAM:
            optimizer = RAdam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.c2)
        elif self.optimizer == OptimizerType.MADGRAD:
            optimizer = Madgrad_wd(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.c2
            )
        else:
            raise AttributeError("Invalid optimizer")

        scheduler = ReduceOnPlateauLearningRateScheduler(
            optimizer,
            factor=0.3,
            patience=self.lr_patience,
            cooldown=10,
            threshold=0.01,
            mode="max",
            verbose=True,  # Vamos tentar ser maior que o lookahaed
            min_lr=1e-5,
        )

        if self.use_bucket_sampler and self.batch_size > 1:
            sampler = MaxSizeFillBucketBatchSampler(
                batch_size=self.batch_size,
                padding_noise=0.3,
                max_size=self.max_tokens_per_batch,
            )
            dl = MultiProcessDataLoader(reader, data_path=(tokens, tags), batch_sampler=sampler)
        else:
            dl = MultiProcessDataLoader(
                reader,
                data_path=(tokens, tags),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )
        dl.index_with(self.vocab)

        if validation_tokens:
            dl_validation = MultiProcessDataLoader(
                reader,
                data_path=(validation_tokens, validation_tags),
                batch_size=self.batch_size,
                shuffle=False,
            )
            dl_validation.index_with(self.vocab)
        else:
            dl_validation = None

        logging.warning(
            f"Start a training with {self.max_iterations} max_iterations, using a {self.network} with {self.layers} "
            f"layers , and {self.hidden_dimension} of hidden size."
        )

        trainer_model = BrokaTrainer

        callback = None
        if progress_callback:
            callback = [MyTrainerCallback(progress_callback, total_iterations=self.max_iterations)]

        trainer = trainer_model(
            model=self.model,
            optimizer=optimizer,
            checkpointer=checkpointer,
            # iterator=iterator,
            grad_norm=10.0,
            data_loader=dl,
            validation_data_loader=dl_validation,
            learning_rate_scheduler=scheduler,
            patience=self.patience,
            num_epochs=self.max_iterations,
            cuda_device=cuda_device,
            max_tokens_per_batch=self.max_tokens_per_batch,
            validation_metric=["+f1-measure-overall", "-loss"],
            min_improvement=0.05,
            callbacks=callback,
        )

        # Limpar o cache antes
        torch.cuda.empty_cache()

        train_metrics = trainer.train()
        self.train_metrics = train_metrics

        logging.info("Finished training")

        # Here's how to save the model.
        if self.model_folder:
            self.save()

        # Limpar o cache antes
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        return train_metrics

    def load(self):
        self.vocab = Vocabulary.from_files(self.model_folder / "model_final" / "vocabulary")

        with open(self.model_folder / "model_final" / "model_params.json", "rb") as f:
            self.set_config_state(pickle.load(f))

        # TODO, this model should be persisted in the trained file
        self.model = self.get_model(self.vocab, use_cache=False)

        # path_torched_model = self.model_folder / "model_final" / "traced_encoder.pt"
        #
        # if path_torched_model.exists():
        #     self.model.encoder._module = torch.jit.load(str(path_torched_model))
        #     logging.warning("Loaded TORCHED encoder")
        # else:
        #     logging.warning("Failed to load TORCHED encoder")

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if torch.cuda.is_available():
            self.model.cuda(0)
        else:
            self.model.cpu()

        self.model.extend_embedder_vocab()

        with open(self.model_folder / "model_final" / "model.th", "rb") as f:
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(f, map_location=util.device_mapping(0)))
            else:
                self.model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    def save(self):
        path_modelo_final = self.model_folder / "model_final"
        path_modelo_final.mkdir(parents=True, exist_ok=True)

        # Switch the model to eval model
        self.model.eval()

        # if self.network == LearningType.SRU:
        #     # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        #     traced_encoder = torch.jit.script(self.model.encoder._module)
        #
        #     # Save the TorchScript model
        #     traced_encoder.save(str(path_modelo_final / "traced_encoder.pt"))
        #
        #     logging.warning("Saved the TORCHED encoder")

        with open(path_modelo_final / "model.th", "wb") as f:
            torch.save(self.model.state_dict(), f)

        with open(path_modelo_final / "model_params.json", "wb") as f:
            pickle.dump(self.get_config_state(), f)

        # And the second is the vocabulary.
        self.vocab.save_to_files(self.model_folder / "model_final" / "vocabulary")
        logging.warning("Saved model")

    def get_config_state(self):
        return {
            "max_iterations": self.max_iterations,
            "network": self.network,
            "layers": self.layers,
            "hidden_dimension": self.hidden_dimension,
            "c1": self.c1,
            "c2": self.c2,
            "bidirectional": self.bidirectional,
            "dropout_layers": self.dropout_layers,
            "learning_rate": self.learning_rate,
            "token_limit": self.token_limit,
            "batch_size": self.batch_size,
            "simple_text": self.simple_text,
            "one_hot_layout": self.one_hot_layout,
            "embedding": self.embedding,
            "train_metrics": self.train_metrics,
            "use_char_cnn": self.use_char_cnn,
            "replace_digits": self.replace_digits,
            # Transformers Exclusive
            "positional_encoding": self.positional_encoding,
            "num_attention_heads": self.num_attention_heads,
            "query_dimensions": self.query_dimensions,
        }

    def set_config_state(self, config):
        self.max_iterations = config["max_iterations"]
        self.network = config["network"]
        self.layers = config["layers"]
        self.hidden_dimension = config["hidden_dimension"]
        self.c1 = config["c1"]
        self.c2 = config["c2"]
        self.bidirectional = config["bidirectional"]
        self.dropout_layers = config["dropout_layers"]
        self.learning_rate = config["learning_rate"]
        self.token_limit = config["token_limit"]
        self.batch_size = config["batch_size"]
        self.simple_text = config["simple_text"]
        self.one_hot_layout = config["one_hot_layout"]
        self.embedding = config["embedding"]
        self.train_metrics = config["train_metrics"]
        self.use_char_cnn = config["use_char_cnn"] if "use_char_cnn" in config else True
        self.replace_digits = config["replace_digits"] if "replace_digits" in config else True

        # Transformers Exclusive
        self.positional_encoding = config["positional_encoding"]
        self.num_attention_heads = config["num_attention_heads"]
        self.query_dimensions = config["query_dimensions"]

        self.set_parameters_from_embedding()

    def is_neural(self):
        return True

    def predict(self, tokens):

        if type(tokens[0]) == list:
            return [self.predict_single(X_single) for X_single in tokens]
        else:
            return self.predict_single(tokens)

    def predict_single(self, tokens):
        predictor, reader = self.prepare_predict()
        # Todo , speedup this by not reloading the predictor for every prediction
        instances = reader.token_to_instances_to_prediction(tokens)
        prediction = predictor.predict_structured_json(instances)
        return prediction

    def prepare_predict(self):
        if self.model is None:
            self.load()
        # Disable unknown tokens error
        logging.getLogger("allennlp.data.vocabulary").setLevel(logging.CRITICAL)
        logging.getLogger("allennlp.common.params").setLevel(logging.CRITICAL)
        logging.getLogger("allennlp.nn.initializers").setLevel(logging.CRITICAL)

        # Disable cache
        self.model.text_field_embedder.disable_cache()

        reader = BrokaReader(
            limit=self.token_limit,
            simple_text=self.simple_text,
            one_hot_layout=self.one_hot_layout,
            use_elmo=self.use_elmo,
            use_flair=self.use_flair,
            use_bert=self.use_bert,
            use_flair_backward=self.use_flair_backward,
            flair_fw_embedding_path=self.flair_fw_embedding_path,
            flair_bw_embedding_path=self.flair_bw_embedding_path,
            use_char_cnn=self.use_char_cnn,
        )
        # predictor = Predictor(self.model.eval(), dataset_reader=reader)
        # TODO - Make this multilingual
        predictor = MultiOIEOpenIePredictor(
            self.model.eval(), dataset_reader=reader, language="pt_core_news_lg"
        )

        return predictor, reader


if __name__ == "__main__":
    pass
