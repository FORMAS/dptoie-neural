import torch.nn.functional as F

from overrides import overrides

import torch

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


class MyMemTransformer(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        network_type,
        feedforward_hidden_dim: int,
        num_attention_heads: int = 8,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        from x_transformers import Encoder

        self.output_size = input_dim

        self.hidden_size = input_dim  # hidden size (i.e. model size)

        self.network_type = network_type

        self._transformer = Encoder(
            dim=input_dim,
            depth=num_layers,
            heads=num_attention_heads,
            use_scalenorm=False,
            ff_glu=True,
            use_rezero=False,
            alibi_pos_bias=True,
            attn_dropout=dropout_prob,  # dropout post-attention
            ff_dropout=dropout_prob,  # feedforward dropout
        )

        self._input_dim = input_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_size

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor):

        output = inputs

        output = self._transformer(output, mask=mask)

        # output = self._transformer(output)

        return output
