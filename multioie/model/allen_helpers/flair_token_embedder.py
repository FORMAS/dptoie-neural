"""
Flair Embedder.
"""
import logging
from hashlib import sha1
from typing import List, Tuple

import flair
import torch
from allennlp.modules.span_extractors import EndpointSpanExtractor
from diskcache import Cache
from flair.embeddings import FlairEmbeddings
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

from multioie.utils.token_iter import gen_split_overlap

logger = logging.getLogger(__name__)
EMBEDDING_VERSION = 3

if torch.cuda.is_available():
    WINDOW = 6144
    OVERLAP = 614
else:
    WINDOW = 4096
    OVERLAP = 400


class FlairEmbedder(TokenEmbedder):
    def __init__(
        self, flair_model: FlairEmbeddings, caching=False, caching_folder=None, batch_size=8
    ) -> None:
        super().__init__()

        self.flair_model = flair_model
        self.pretrain_name = self.flair_model.name
        self.output_dim = flair_model.lm.hidden_size
        self.chars_per_chunk = 512
        self.WINDOW = WINDOW
        self.OVERLAP = OVERLAP
        self.batch_size = batch_size
        self.caching = caching
        if caching:
            self.cache: Cache = Cache(caching_folder, size_limit=2 ** 34)  # 16G of storage
            self.cache.cull()
        else:
            self.cache = None
            if caching_folder:
                Cache(caching_folder, size_limit=2 ** 34).expire()

        for param in self.flair_model.lm.parameters():
            param.requires_grad = False

        # In Flair, every LM is unidirectional going forwards.
        # We always extract on the right side.
        comb_string = "y"

        self.span_extractor = EndpointSpanExtractor(
            input_dim=self.flair_model.lm.hidden_size, combination=comb_string
        )

        # Set model to None so it is reloaded in the forward.
        # Have no idea what is happening, but something is modifying the model
        # somewhere between init and forward. Reloading in forward works.
        # self.flair_model = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self, tokens: torch.LongTensor, spans: torch.LongTensor, flair_mask: torch.BoolTensor
    ) -> torch.Tensor:

        if self.cache:
            self.cache.cull()

        with torch.no_grad():
            # Doesn't matter what this key is, just needs to be a dict.
            # mask = get_text_field_mask({"chars": tokens})

            # trick: fake spans have a sum that is 0 or less.
            # Look at FlairCharIndexer.pad_token_sequence() to see what the default tuple value is
            mask_spans = (spans.sum(dim=2) > 0).long()

            output = self.multiple_cached_forward(flair_mask, mask_spans, spans, tokens)

        return torch.nn.utils.rnn.pad_sequence(output, batch_first=True)

    def multiple_cached_forward(self, flair_mask, mask_spans, spans, tokens):
        output = dict()

        batches: List[torch.Tensor] = []
        pos_of_batches: List[Tuple[int, str, torch.Tensor]] = []
        pos_of_split = dict()

        # TODO: This should use a vector of batch size to process the hidden as the same size of the batch
        for pos, batch_element in enumerate(tokens):
            # Truncate the padding to more efficiently process batches
            max_span_tensor, max_span_pos = torch.max(spans[pos], 0)
            max_pos_span = max_span_tensor[1]
            batch_element = batch_element[: max_pos_span + 1]
            trucate_span_pos = int((mask_spans[pos] == 1).sum(dim=0))
            truncated_spans = spans[pos][: trucate_span_pos + 1]

            size_batch = len(batch_element)

            key = None
            if self.caching:
                key = (
                    f"{size_batch}-{EMBEDDING_VERSION}-{self.pretrain_name}-{sha1(batch_element.cpu().numpy().tobytes()).hexdigest()}-"
                    f"{sha1(truncated_spans.cpu().numpy().tobytes()).hexdigest()}"
                )
                if key in self.cache:
                    output[pos] = self.cache[key]
                    continue

            # If we have less than the batch size, lets split big sentences
            if (len(batch_element) > (self.WINDOW * 2)) and (len(tokens) < self.batch_size):
                splited_element = self.split_huge_embeddings(batch_element)
                pos_of_split[pos] = len(splited_element)
                for el in splited_element:
                    batches.append(el)
                    pos_of_batches.append((pos, key, truncated_spans))
            else:
                batches.append(batch_element)
                pos_of_batches.append((pos, key, truncated_spans))

        # Quebrar em batchs
        if len(batches) > 0:
            rnn_outputs = self.get_embeddings(batches)

            merged_output = []
            merged_pos_of_batches = []
            i_output = 0
            while i_output < len(rnn_outputs):
                half_overlap = self.OVERLAP // 2
                pos, _, _ = pos_of_batches[i_output]
                first_i = i_output
                if pos in pos_of_split:
                    merge_array = []
                    size_to_merge = pos_of_split[pos]

                    for i_merge in range(size_to_merge):
                        first = i_merge == 0
                        last = i_merge == size_to_merge - 1

                        if not first and not last:
                            merge_array.append(rnn_outputs[i_merge][half_overlap:-half_overlap])
                        elif first and last:
                            merge_array.append(rnn_outputs[i_merge])
                        elif last:
                            merge_array.append(rnn_outputs[i_merge][half_overlap:])
                        else:
                            merge_array.append(rnn_outputs[i_merge][:-half_overlap])
                    # Finally increment the actual position
                    i_output += size_to_merge - 1

                    merged_output.append(torch.cat(merge_array))
                    merged_pos_of_batches.append(pos_of_batches[first_i])
                else:
                    merged_output.append(rnn_outputs[i_output])
                    merged_pos_of_batches.append(pos_of_batches[i_output])
                i_output += 1

            del rnn_outputs
            pos_of_batches = merged_pos_of_batches

            if len(pos_of_batches) != len(merged_output):
                raise ValueError("Different sizes")

            for pos, rnn_output in zip(pos_of_batches, merged_output):
                pos, key, truncated_spans = pos
                size_truncated = len(truncated_spans)
                word_embeddings = (
                    self.span_extractor(
                        rnn_output.contiguous().unsqueeze(0),
                        truncated_spans.unsqueeze(0),
                        flair_mask[pos][:size_truncated].unsqueeze(0),
                        mask_spans[pos][:size_truncated].unsqueeze(0),  # type: ignore
                    )
                    .contiguous()
                    .squeeze(0)
                )
                output[pos] = word_embeddings
                if self.caching and key is not None:
                    self.cache.set(
                        key, word_embeddings, expire=60 * 60 * 24 * 5
                    )  # 5 days to expire

            del merged_output

        return [output[x] for x in range(len(tokens))]

    def remerge_splited(self, batches):
        HALF_OVERLAP = self.OVERLAP // 2
        total_size = len(batches)
        results = []
        for i, batch in enumerate(batches):
            first = i == 0
            last = i == total_size - 1
            if not first and not last:
                results.extend(batches[HALF_OVERLAP:-HALF_OVERLAP])
            elif first and last:
                results.extend(batches)
            elif last:
                results.extend(batches[HALF_OVERLAP:])
            else:
                results.extend(batches[:-HALF_OVERLAP])
        return results

    def split_huge_embeddings(self, batch_element):

        partes = []
        for slice, _, _ in gen_split_overlap(batch_element, self.WINDOW, self.OVERLAP):
            partes.append(slice)
        return partes

    def get_embeddings(self, batch_element):

        longest_padded_str: int = len(max(batch_element, key=len))

        # cut up the input into chunks of max charlength = chunk_size
        chunks = []
        splice_begin = 0

        for splice_end in range(self.chars_per_chunk, longest_padded_str, self.chars_per_chunk):
            chunks.append([text[splice_begin:splice_end] for text in batch_element])
            splice_begin = splice_end

        chunks.append([text[splice_begin:longest_padded_str] for text in batch_element])

        hidden = self.flair_model.lm.init_hidden(len(chunks[0]))

        padding_char_index = self.flair_model.lm.dictionary.get_idx_for_item(" ")
        batches: List[torch.Tensor] = []
        # push each chunk through the RNN language model
        for chunk in chunks:
            t = torch.nn.utils.rnn.pad_sequence(
                chunk, batch_first=True, padding_value=padding_char_index
            ).to(device=flair.device, non_blocking=True)
            batches.append(t)

        output_parts = []
        for batch in batches:
            batch = batch.transpose(0, 1)
            _, rnn_output, hidden = self.flair_model.lm.forward(batch, hidden)
            output_parts.append(rnn_output)

        result = torch.cat(output_parts).transpose(0, 1)
        return result


@TokenEmbedder.register("flair-pretrained")
class PretrainedFlairEmbedder(FlairEmbedder):
    # pylint: disable=line-too-long
    """
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'news-forward'),

        If the name is a key in the list of pretrained models at
        https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py#L834
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    """

    def __init__(self, pretrained_model: str) -> None:
        flair_embs = FlairEmbeddings(pretrained_model)

        super().__init__(flair_model=flair_embs)
