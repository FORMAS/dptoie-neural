import re
import unicodedata


class Featurizer:
    def __init__(self):
        self.last_p_id = None
        self.last_p_bottom = None

    def rule_to_value(self, rule_value):
        try:
            rule_value = rule_value.replace(";", "").strip()

            ends_with_pt = True if rule_value.endswith("pt") else False

            int_value = int("".join(list(filter(str.isdigit, rule_value))))

            if ends_with_pt:
                int_value *= 1.333

            return int(int_value)
        except:  # noqa
            return 0

    def css_rules_to_dimensions(self, rule):
        top = 0
        right = 0
        bottom = 0
        left = 0

        pieces = rule.split(" ")

        if len(pieces) == 1:
            top = right = bottom = left = self.rule_to_value(pieces[0])
        elif len(pieces) == 2:
            top = self.rule_to_value(pieces[0])
            right = self.rule_to_value(pieces[1])
        elif len(pieces) == 3:
            top = self.rule_to_value(pieces[0])
            right = self.rule_to_value(pieces[1])
            bottom = self.rule_to_value(pieces[2])
        elif len(pieces) == 4:
            top = self.rule_to_value(pieces[0])
            right = self.rule_to_value(pieces[1])
            bottom = self.rule_to_value(pieces[2])
            left = self.rule_to_value(pieces[3])

        return top, right, bottom, left

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

        # TODO - Use a real feature for ONE hot feature
        result["dummy_one_hot"] = "1"

        return result
