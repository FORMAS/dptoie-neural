import re
import unicodedata


class Featurizer:
    def __init__(self):
        self.last_p_id = None
        self.last_p_bottom = None

    def agregado(self, html_token, replace_with_number=True):
        # Listar todas as features possiveis
        token = html_token.token.rstrip()

        # no_accents_token
        lower = token.lower()
        no_accents = "".join(
            (c for c in unicodedata.normalize("NFD", lower) if unicodedata.category(c) != "Mn")
        )

        # word_shape
        if len(token) >= 100:
            word_shape = "LONG"
            word_shape_degen = "LONG"
        else:
            shape = []
            shape_degenerate = []
            last = ""
            shape_char = ""
            seq = 0
            for c in token:
                if c.isalpha():
                    if c.isupper():
                        shape_char = "X"
                    else:
                        shape_char = "x"
                elif c.isdigit():
                    shape_char = "d"
                else:
                    shape_char = c
                if shape_char == last:
                    seq += 1
                else:
                    seq = 0
                    last = shape_char
                if seq < 4:
                    shape.append(shape_char)
                    if seq == 0:
                        shape_degenerate.append(shape_char)
            word_shape = "".join(shape)
            word_shape_degen = "".join(shape_degenerate)

        if replace_with_number:
            token = re.sub(r"\d", "0", token)
            no_accents = re.sub(r"\d", "0", no_accents)

        result = dict()

        # Definir as features
        result["token"] = token

        result["no_accents"] = no_accents

        result["word_shape"] = word_shape
        result["word_shape_degen"] = word_shape_degen

        result["pos"] = html_token.pos
        result["dep"] = html_token.dep

        # TODO - Use a real feature for ONE hot feature
        result["dummy_one_hot"] = "1"

        return result
