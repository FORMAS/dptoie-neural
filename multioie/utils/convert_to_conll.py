# Portuguese dataset
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from allennlp.data.tokenizers import SpacyTokenizer
from allennlp_models.structured_prediction.models.srl import convert_bio_tags_to_conll_format

from multioie.utils.contractions import transform_portuguese_contractions, clean_extraction

tokenizer = SpacyTokenizer(language="pt_core_news_lg", split_on_spaces=True)


def find_sublist_match(tokens: List, string_to_find: str, start=0):
    # TODO implement a Needleman-Wunsch or Smith-Waterman algorithms
    start_match = 0
    end_match = 0

    if len(string_to_find) < 1:
        return None, None

    if start is None:
        start = 0

    string_to_find = tokenizer.tokenize(string_to_find)

    first_to_find = string_to_find[0]
    last_to_find = string_to_find[-1]

    achou = False
    while not achou and (len(tokens) - start) > len(string_to_find):

        for pos in range(start, len(tokens)):
            if first_to_find.text == tokens[pos].text:
                # print(f"Achei {first_to_find} na pos {pos}")
                start_match = pos

                for pos_fim in range(pos, len(tokens)):
                    if (last_to_find.text == tokens[pos_fim].text) and (
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
        return None, None
    return start_match, end_match


def convert_to_conll(sentence, pos_sentence):
    frase = sentence["phrase"]

    # Find all verbs in the input sentence
    spacy_tokens = tokenizer.tokenize(frase)

    pred_ids = [i for (i, t) in enumerate(spacy_tokens) if t.pos_ in ["VERB", "AUX"]]

    sentence["broken"] = 0
    sentence["correct"] = 0

    for extracao in sentence["extractions"]:
        start_pos_arg1, end_pos_arg1 = find_sublist_match(spacy_tokens, extracao["arg1"])
        start_pos_rel, end_pos_rel = find_sublist_match(spacy_tokens, extracao["rel"], end_pos_arg1)
        start_pos_arg2, end_pos_arg2 = find_sublist_match(
            spacy_tokens, extracao["arg2"], end_pos_rel
        )

        if any(elem is None for elem in [start_pos_arg1, start_pos_rel, start_pos_arg2]):
            print(
                f"Elemento quebrado: {spacy_tokens} - {extracao['arg1']=} - {extracao['rel']=} - {extracao['arg2']=}"
            )
            sentence["broken"] += 1
        else:
            sentence["correct"] += 1

            tags = []

            for pos in range(len(spacy_tokens)):
                if pos == start_pos_arg1:
                    tags.append("B-ARG0")
                elif pos == start_pos_rel:
                    tags.append("B-V")
                elif pos == start_pos_arg2:
                    tags.append("B-ARG1")

                elif end_pos_arg1 >= pos > start_pos_arg1:
                    tags.append("I-ARG0")
                elif end_pos_arg2 >= pos > start_pos_arg2:
                    tags.append("I-ARG1")
                elif end_pos_rel >= pos > start_pos_rel:
                    tags.append("I-V")
                else:
                    tags.append("O")

            convertido = convert_bio_tags_to_conll_format(tags)

            result = []
            for idx, token in enumerate(frase.split(" ")):
                actual_tag = convertido[idx]
                line = f"{pos_sentence}\t{idx}\t{token}\tXX\t-\t-\t-\t-\t-\t*\t{actual_tag}\t-"

                result.append(line)

            extracao["conll"] = result
    # Set a empty tag if the sentence is empty
    result = []
    for idx, token in enumerate(frase.split(" ")):
        line = f"{pos_sentence}\t{idx}\t{token}\tXX\t-\t-\t-\t-\t-\t*\tO\t-"
        result.append(line)

    sentence["empty_tags"] = result


def load_pragmatic_wiki_dataset():
    dataset_pt = dict()

    pt = Path("../../datasets/pragmatic_dataset/wiki200.txt")
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            line = line.strip()
            pos, phrase = line.split("\t", 1)
            phrase = clean_extraction(phrase)
            dataset_pt[int(pos)] = {
                "phrase": transform_portuguese_contractions(phrase.strip()),
                "extractions": [],
            }

    pt = Path("../../datasets/pragmatic_dataset/wiki200-labeled.csv")
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            if "\t" in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = clean_extraction(partes[1])
                rel = clean_extraction(partes[2])
                arg2 = clean_extraction(partes[3])
                valid = partes[-1].strip()

                if valid != "1":
                    continue  # so queremos pegar as positivas

                dataset_pt[pos]["extractions"].append(
                    {
                        "arg1": transform_portuguese_contractions(arg1),
                        "rel": transform_portuguese_contractions(rel),
                        "arg2": transform_portuguese_contractions(arg2),
                        "valid": int(valid),
                    }
                )
    return dataset_pt


def load_pragmatic_ceten_dataset():
    dataset_pt = dict()

    pt = Path("../../datasets/pragmatic_dataset/ceten200.txt")
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            line = line.strip()
            pos, phrase = line.split("\t", 1)
            phrase = clean_extraction(phrase)
            dataset_pt[int(pos)] = {
                "phrase": transform_portuguese_contractions(phrase.strip()),
                "extractions": [],
            }

    pt = Path("../../datasets/pragmatic_dataset/ceten200-labeled.csv")
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            if "\t" in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = clean_extraction(partes[1])
                rel = clean_extraction(partes[2])
                arg2 = clean_extraction(partes[3])
                valid = partes[-1].strip()

                if valid != "1":
                    continue  # so queremos pegar as positivas

                dataset_pt[pos]["extractions"].append(
                    {
                        "arg1": transform_portuguese_contractions(arg1),
                        "rel": transform_portuguese_contractions(rel),
                        "arg2": transform_portuguese_contractions(arg2),
                        "valid": int(valid),
                    }
                )
    return dataset_pt


def load_pud200():
    # Portuguese dataset
    dataset_pt = dict()
    for name in ["200-sentences-pt-PUD.txt"]:
        pt = Path(f"../../datasets/coling_PUD/{name}")
        with open(pt, "r", encoding="utf-8") as f_pt:
            actual_pos = None
            for line in f_pt:
                line = line.strip()
                pos, phrase = line.split("\t", 1)
                phrase = clean_extraction(phrase)

                if pos.isnumeric() and phrase.count("\t") < 1:
                    actual_pos = int(pos)
                    phrase = transform_portuguese_contractions(phrase)
                    # phrase = re.sub(r',|\.|"', "", phrase)
                    dataset_pt[actual_pos] = {"phrase": phrase, "extractions": []}
                else:
                    partes = line.split("\t")
                    arg1 = clean_extraction(partes[0])
                    rel = clean_extraction(partes[1])
                    arg2 = clean_extraction(partes[2])
                    valid = int(partes[-2].strip())

                    if valid != 1:
                        continue  # so queremos pegar as positivas

                    dataset_pt[actual_pos]["extractions"].append(
                        {
                            "arg1": transform_portuguese_contractions(arg1),
                            "rel": transform_portuguese_contractions(rel),
                            "arg2": transform_portuguese_contractions(arg2),
                            "valid": valid,
                        }
                    )

    return dataset_pt




def load_pud100():
    # Portuguese dataset
    dataset_pt = dict()
    for name in ["coling2020.txt"]:
        pt = Path(f"../../datasets/coling_PUD/{name}")
        with open(pt, "r", encoding="utf-8") as f_pt:
            actual_pos = None
            for line in f_pt:
                line = line.strip()
                pos, phrase = line.split("\t", 1)
                # phrase = clean_extraction(phrase)

                if pos.isnumeric() and phrase.count("\t") < 1:
                    actual_pos = int(pos)
                    # phrase = transform_portuguese_contractions(phrase)
                    # phrase = re.sub(r',|\.|"', "", phrase)
                    dataset_pt[actual_pos] = {"phrase": phrase, "extractions": []}
                else:
                    partes = line.split("\t")
                    arg1 = clean_extraction(partes[0])
                    rel = clean_extraction(partes[1])
                    arg2 = clean_extraction(partes[2])
                    valid = int(partes[-2].strip())

                    if valid != 1:
                        continue  # so queremos pegar as positivas

                    dataset_pt[actual_pos]["extractions"].append(
                        {
                            "arg1": transform_portuguese_contractions(arg1),
                            "rel": transform_portuguese_contractions(rel),
                            "arg2": transform_portuguese_contractions(arg2),
                            "valid": valid,
                        }
                    )

    return dataset_pt


def load_anderson():

    files_path = Path(f"../../datasets/anderson").resolve()
    files = files_path.glob('*')

    dataset_pt = dict()
    actual_pos = 0

    for pt in files:
        with open(pt, "r", encoding="utf-8") as f_pt:
            for line in f_pt:

                result = json.loads(line)

                pos, phrase = line.split("\t", 1)
                # phrase = clean_extraction(phrase)

                if pos.isnumeric() and phrase.count("\t") < 1:
                    actual_pos = int(pos)
                    # phrase = transform_portuguese_contractions(phrase)
                    # phrase = re.sub(r',|\.|"', "", phrase)
                    dataset_pt[actual_pos] = {"phrase": phrase, "extractions": []}
                else:
                    partes = line.split("\t")
                    arg1 = clean_extraction(partes[0])
                    rel = clean_extraction(partes[1])
                    arg2 = clean_extraction(partes[2])
                    valid = int(partes[-2].strip())

                    if valid != 1:
                        continue  # so queremos pegar as positivas

                    dataset_pt[actual_pos]["extractions"].append(
                        {
                            "arg1": transform_portuguese_contractions(arg1),
                            "rel": transform_portuguese_contractions(rel),
                            "arg2": transform_portuguese_contractions(arg2),
                            "valid": valid,
                        }
                    )

    return dataset_pt


def load_gamalho():
    # Portuguese dataset
    dataset_pt = dict()

    pt = Path("../../datasets/gamalho/sentences.txt")
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            line = line.strip()
            pos, phrase = line.split("\t", 1)
            phrase = clean_extraction(phrase)
            dataset_pt[int(pos)] = {
                "phrase": transform_portuguese_contractions(phrase),
                "extractions": [],
            }

    pt = Path("../../datasets/gamalho/gold.csv")

    def clean_at_symbol(text):
        text = text.replace("@", " ")
        return text

    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            if "\t" in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = clean_at_symbol(clean_extraction(partes[1]))
                rel = clean_at_symbol(clean_extraction(partes[2]))
                arg2 = clean_at_symbol(clean_extraction(partes[3]))
                if len(partes[-1].strip()) < 1:
                    continue
                valid = int(partes[-1].strip())

                if valid != 1:
                    continue  # so queremos pegar as positivas

                dataset_pt[pos]["extractions"].append(
                    {
                        "arg1": transform_portuguese_contractions(arg1),
                        "rel": transform_portuguese_contractions(rel),
                        "arg2": transform_portuguese_contractions(arg2),
                        "valid": valid,
                    }
                )

    return dataset_pt


def save_data_as_conll(pos_dataset, output_path,dataset_to_save):
    with open(output_path
            , "w", encoding="utf-8"
              ) as f_out:
        for sentence in dataset_to_save:
            if len(sentence["extractions"]) < 1:

                #if sentence['broken'] > 0:  # We going to skip if it is our fault
                #    continue
                for line in sentence["empty_tags"]:
                    f_out.write(f"train/train/{pos_dataset}\t{line}\n")
                f_out.write("\n")

            else:
                for extraction in sentence["extractions"]:
                    if "conll" not in extraction:
                        for line in sentence["empty_tags"]:
                            f_out.write(f"train/train/{pos_dataset}\t{line}\n")
                        f_out.write("\n")
                        continue

                    for line in extraction["conll"]:
                        f_out.write(f"train/train/{pos_dataset}\t{line}\n")
                    f_out.write("\n")
        f_out.write("\n")


if __name__ == "__main__":
    global qt_quebrados, qt_funfando
    datasets = [
        {"name": "pragmatic_wiki", "dataset": load_pragmatic_wiki_dataset(), "broken": 0, "correct": 0},
        {"name": "pragmatic_ceten", "dataset": load_pragmatic_ceten_dataset(), "broken": 0, "correct": 0},
        {"name": "gamalho", "dataset": load_gamalho(), "broken": 0, "correct": 0},
        {"name": "pud_200", "dataset": load_pud200(), "broken": 0, "correct": 0},
        {"name": "pud_100", "dataset": load_pud100(), "broken": 0, "correct": 0},
        #{"name": "anderson", "dataset": load_anderson(), "broken": 0, "correct": 0},
    ]

    actual_pos = 0

    for dataset_to_process in datasets:

        extractions = dataset_to_process["dataset"]

        for pos_sentence, value in enumerate(extractions.values()):
            #print(value["phrase"])
            convert_to_conll(value, pos_sentence)
            dataset_to_process["broken"] += value["broken"]
            dataset_to_process["correct"] += value["correct"]

    for pos_dataset, dataset_to_process in enumerate(datasets):
        print(dataset_to_process["name"])
        qt_quebrados = dataset_to_process["broken"]
        qt_funfando = dataset_to_process["correct"]
        print("Qtd de elementos quebrados: ", qt_quebrados)
        print("Qtd de elementos funfando: ", qt_funfando)
        print("Qtd de elementos totais: ", qt_quebrados + qt_funfando)
        print(f"Sentences : {len(dataset_to_process['dataset'])}")

        output_path = f"../../datasets/meu_dataset_round2/{dataset_to_process['name']}.conll"
        dataset_to_save = dataset_to_process["dataset"].values()
        save_data_as_conll(pos_dataset, output_path, dataset_to_save)
