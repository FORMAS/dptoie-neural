from typing import Dict, List, Optional

import numpy
import torch
from allennlp.common.util import pad_sequence_to_length
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList


class ArrayIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as single integers.

    Registered as a `TokenIndexer` with name "single_id".

    # Parameters

    namespace : `Optional[str]`, optional (default=`tokens`)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.  If you
        explicitly pass in `None` here, we will skip indexing and vocabulary lookups.  This means
        that the `feature_name` you use must correspond to an integer value (like `text_id`, for
        instance, which gets set by some tokenizers, such as when using byte encoding).
    feature_name : `str`, optional (default=`text`)
        We will use the :class:`Token` attribute with this name as input.  This is potentially
        useful, e.g., for using NER tags instead of (or in addition to) surface forms as your inputs
        (passing `ent_type_` here would do that).  If you use a non-default value here, you almost
        certainly want to also change the `namespace` parameter, and you might want to give a
        `default_value`.
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        namespace: Optional[str] = "tokens",
        feature_name: str = "text",
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        self.namespace = namespace

        self._feature_name = feature_name

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        return

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in tokens:
            text = self._get_feature_value(token)
            if self.namespace is None:
                # We could have a check here that `text` is an int; not sure it's worth it.
                indices.append(text)  # type: ignore
            else:
                indices.append(text)  # type: ignore

        return {"tokens": indices}

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}

    def _get_feature_value(self, token: Token) -> str:
        text = getattr(token, self._feature_name)
        if text is None:
            raise ValueError(
                f"{token} did not have attribute {self._feature_name}. If you "
                "want to ignore this kind of error, give a default value in the "
                "constructor of this indexer."
            )
        return text

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:

        # All arrays have the same size
        array_size = len(tokens["tokens"][0])

        def padding_token():
            return numpy.zeros(array_size, dtype=numpy.float32)

        tensor = torch.FloatTensor(
            pad_sequence_to_length(
                tokens["tokens"], padding_lengths["tokens"], default_value=padding_token
            )
        )
        return {"tokens": tensor}
