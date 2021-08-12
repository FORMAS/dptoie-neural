import torch
from allennlp.modules import Seq2SeqEncoder
from overrides import overrides


class MySRUWrapper(Seq2SeqEncoder):
    def __init__(self, module: torch.nn.Module, srupp=False) -> None:
        super().__init__()
        self._module = module
        self.srupp = srupp

        try:
            self._is_bidirectional = self._module.bidirectional
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    @overrides
    def get_input_dim(self) -> int:
        return self._module.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self._module.hidden_size * self._num_directions

    @overrides
    def is_bidirectional(self) -> bool:
        return self._is_bidirectional

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor):
        # inputs it is in the format (batch, sequence, features)
        if self.srupp:
            output, hidden_state, _ = self._module(
                inputs.permute(1, 0, 2), mask_pad=(~mask).permute(1, 0)
            )
        else:
            output, hidden_state = self._module(
                inputs.permute(1, 0, 2), mask_pad=(~mask).permute(1, 0)
            )
        return output.permute(1, 0, 2)
