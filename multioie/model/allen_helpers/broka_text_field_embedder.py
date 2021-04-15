from typing import Dict
import inspect

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


class BrokaTextFieldEmbedder(TextFieldEmbedder):
    """
    This is a `TextFieldEmbedder` that wraps a collection of
    [`TokenEmbedder`](../token_embedders/token_embedder.md) objects.  Each
    `TokenEmbedder` embeds or encodes the representation output from one
    [`allennlp.data.TokenIndexer`](../../data/token_indexers/token_indexer.md). As the data produced by a
    [`allennlp.data.fields.TextField`](../../data/fields/text_field.md) is a dictionary mapping names to these
    representations, we take `TokenEmbedders` with corresponding names.  Each `TokenEmbedders`
    embeds its input, and the result is concatenated in an arbitrary (but consistent) order.

    # Parameters


    token_embedders : `Dict[str, TokenEmbedder]`, required.
        A dictionary mapping token embedder names to implementations.
        These names should match the corresponding indexer used to generate
        the tensor passed to the TokenEmbedder.
    """

    def __init__(
        self,
        token_embedders: Dict[str, TokenEmbedder],
        one_hot_size: int,
        use_separated_one_hot=False,
    ) -> None:
        super().__init__()
        # NOTE(mattg): I'd prefer to just use ModuleDict(token_embedders) here, but that changes
        # weight locations in torch state dictionaries and invalidates all prior models, just for a
        # cosmetic change in the code.
        self._token_embedders = token_embedders
        self._one_hot_size = one_hot_size
        for key, embedder in token_embedders.items():
            name = "token_embedder_%s" % key
            self.add_module(name, embedder)
        self._ordered_embedder_keys = sorted(self._token_embedders.keys())
        self.use_separated_one_hot = use_separated_one_hot

    def disable_cache(self):
        if "flair" in self._token_embedders:
            self._token_embedders["flair"].caching = False

    @overrides
    def get_output_dim(self) -> int:
        output_dim = self._one_hot_size if (not self.use_separated_one_hot) else 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(
        self, text_field_input: TextFieldTensors, num_wrapping_dims: int = 0, **kwargs
    ) -> torch.Tensor:

        actual_keys = set(text_field_input.keys())
        have_one_hot = False
        if (not self.use_separated_one_hot) and "one_hot" in actual_keys:
            actual_keys.remove("one_hot")
            have_one_hot = True

        if self._token_embedders.keys() != actual_keys:
            message = "Mismatched token keys: %s and %s" % (
                str(self._token_embedders.keys()),
                str(text_field_input.keys()),
            )
            raise ConfigurationError(message)

        embedded_representations = []

        # Adicionar onehot
        if (not self.use_separated_one_hot) and have_one_hot:
            embedded_representations.append(text_field_input["one_hot"]["tokens"].float())

        for key in self._ordered_embedder_keys:
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, "token_embedder_{}".format(key))
            forward_params = inspect.signature(embedder.forward).parameters
            forward_params_values = {}
            missing_tensor_args = set()
            for param in forward_params.keys():
                if param in kwargs:
                    forward_params_values[param] = kwargs[param]
                else:
                    missing_tensor_args.add(param)

            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)

            tensors: Dict[str, torch.Tensor] = text_field_input[key]
            if len(tensors) == 1 and len(missing_tensor_args) == 1:
                # If there's only one tensor argument to the embedder, and we just have one tensor to
                # embed, we can just pass in that tensor, without requiring a name match.
                token_vectors = embedder(list(tensors.values())[0], **forward_params_values)
            else:
                # If there are multiple tensor arguments, we have to require matching names from the
                # TokenIndexer.  I don't think there's an easy way around that.
                token_vectors = embedder(**tensors, **forward_params_values)
            if token_vectors is not None:
                # To handle some very rare use cases, we allow the return value of the embedder to
                # be None; we just skip it in that case.
                embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)
