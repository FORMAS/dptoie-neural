import re


def find_sublist_match(tokens, string_to_find, start=0):
    start_match = 0
    end_match = 0

    if len(string_to_find) < 1:
        print("Vazio!")
        return 0, 0

    first_to_find = string_to_find[0]
    last_to_find = string_to_find[-1]

    achou = False
    while not achou and (len(tokens) - start) > len(string_to_find):

        for pos in range(start, len(tokens)):
            if first_to_find == tokens[pos]:
                # print(f"Achei {first_to_find} na pos {pos}")
                start_match = pos

                for pos_fim in range(pos, len(tokens)):
                    if (last_to_find == tokens[pos_fim]) and (
                        pos_fim - start_match >= len(string_to_find) - 2
                    ):
                        # print(f"Achei o FIM {last_to_find} na pos {pos_fim}")
                        end_match = pos_fim
                        achou = True
                        break
                if (len(string_to_find) == 1 or end_match > 0) and (
                    (abs((end_match - start_match) - len(string_to_find)) < 20)
                    or ((end_match - start_match) > len(string_to_find))
                ):
                    break
        if achou:
            break
        start += 1

    if end_match == 0 and start_match > 0:
        print(f"NÃO Encontrei {string_to_find}")
    return start_match, end_match


def clean_extraction(text):
    text = text.strip()
    text = text.strip('"')
    text = text.strip(".")
    text = text.strip()
    text = text.replace(",", " , ")
    text = text.replace(" ", " ")
    text = text.replace("\u200b", "")
    text = " ".join(filter(None, text.split(" ")))
    text = " ".join(filter(None, text.split("_")))
    return text


def transform_portuguese_contractions(texto):
    contracoes = {
        "do": "de o",
        "da": "de a",
        "no": "em o",
        "na": "em a",
        "dos": "de os",
        "ao": "a o",
        "das": "de as",
        "à": "a a",
        "pelo": "por o",
        "pela": "por a",
        "nos": "em os",
        "aos": "a os",
        "nas": "em as",
        # "às": "a as",
        "dum": "de um",
        "duma": "de umas",
        # "pelos": "por os",
        "num": "em um",
        "numa": "em uma",
        # "pelas": "por as",
        # "doutros": "de outros",
        "nalguns": "em alguns",
        "dalguns": "de alguns",
        "noutras": "em outras",
        "dalgumas": "de algumas",
        "doutra": "de outra",
        "noutros": "em outros",
        "nalgumas": "em algumas",
        "doutras": "de outras",
        "noutro": "em outro",
        "donde": "de onde",
        "doutro": "de outro",
        "noutra": "em outra",
        "dalguma": "de alguma",
        "dalgum": "de algum",
        "dalguém": "de alguém",
    }
    texto_convertido = texto
    for nome, replacement in contracoes.items():
        texto_convertido = re.sub(fr"\b{nome}\b", replacement, texto_convertido)

    return " ".join(filter(None, texto_convertido.split(" ")))
