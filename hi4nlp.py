import gc
import os
import pickle
import re
import shutil
import logging
import argparse
import tarfile
from pathlib import Path
import random

import numpy as np

from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier, DecisionListClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.experimental import enable_hist_gradient_boosting

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models.archival import archive_model, load_archive
from allennlp.predictors import Predictor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, matthews_corrcoef

from mllp.models import MLLP
from udify import util

from pathlib import Path


def carregar_lrec():
    # Portuguese dataset
    dataset_pt = dict()

    pt = Path("Dataset/lrec/200-sentences-pt-PUD.txt")
    with open(pt, "r", encoding="utf-8") as f_pt:
        actual_pos = None
        for line in f_pt:
            line = line.strip()
            partes = line.split("\t")
            if len(partes) == 2:
                actual_pos = int(partes[0])
                dataset_pt[actual_pos] = {
                    "phase": transformar_contracoes_pt(partes[1].strip()),
                    "extractions": [],
                }
            else:
                partes = line.split("\t")
                arg1 = partes[0].strip('"')
                rel = partes[1].strip('"')
                arg2 = partes[2].strip('"')
                valid = partes[-2]

                dataset_pt[actual_pos]["extractions"].append(
                    {
                        "arg1": transformar_contracoes_pt(arg1),
                        "rel": transformar_contracoes_pt(rel),
                        "arg2": transformar_contracoes_pt(arg2),
                        "valid": valid.strip(),
                    }
                )

    # English dataset
    dataset_en = dict()

    en = Path("Dataset/gamalho/en/sentences.txt")
    with open(en, "r", encoding="utf-8") as f_en:
        for line in f_en:
            pos, phase = line.split("\t")
            dataset_en[int(pos)] = {"phase": phase, "extractions": []}

    en = Path("Dataset/gamalho/en/extractions-all-labeled.txt")
    with open(en, "r", encoding="utf-8") as f_en:
        for line in f_en:
            if "\t" in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = partes[1].strip('"')
                rel = partes[2].strip('"')
                arg2 = partes[3].strip('"')
                valid = partes[-1]

                dataset_en[pos]["extractions"].append(
                    {"arg1": arg1, "rel": rel, "arg2": arg2, "valid": valid.strip()}
                )

        # Spanish dataset
        dataset_es = dict()

        es = Path("Dataset/gamalho/es/sentences.txt")
        with open(es, "r", encoding="utf-8") as f_es:
            for line in f_es:
                line = line.strip()
                pos, phase = line.split("\t", 1)
                dataset_es[int(pos)] = {
                    "phase": transformar_contracoes_es(phase.strip()),
                    "extractions": [],
                }

        es = Path("Dataset/gamalho/es/extractions-all-labeled.txt")
        with open(es, "r", encoding="utf-8") as f_es:
            for line in f_es:
                if "\t" in line:
                    partes = line.split("\t")
                    pos = int(partes[0])
                    arg1 = transformar_contracoes_es(partes[1].strip('"'))
                    rel = transformar_contracoes_es(partes[2].strip('"'))
                    arg2 = transformar_contracoes_es(partes[3].strip('"'))
                    valid = partes[-1]

                    dataset_es[pos]["extractions"].append(
                        {"arg1": arg1, "rel": rel, "arg2": arg2, "valid": valid.strip()}
                    )

    return dataset_en, dataset_pt, dataset_es


def report_performance(docs_en, docs_pt, docs_es):
    from sklearn.metrics import classification_report, matthews_corrcoef

    print("---- English ----")
    y_true_en = []
    for pos, doc in docs_en.items():
        for extraction in doc["extractions"]:
            # if any(len(x) < 1 for x in extraction.values()):
            #     continue

            y_true_en.append(extraction["valid"])
    y_predicted_en = ["1"] * len(y_true_en)
    print(classification_report(y_true_en, y_predicted_en, digits=6))
    print("Matthews EN:")
    print(matthews_corrcoef(y_true_en, y_predicted_en))

    print("---- Portuguese ----")
    y_true_pt = []
    for pos, doc in docs_pt.items():
        for extraction in doc["extractions"]:
            # if any(len(x) < 1 for x in extraction.values()):
            #     continue

            y_true_pt.append(extraction["valid"])
    y_predicted_pt = ["1"] * len(y_true_pt)
    print(classification_report(y_true_pt, y_predicted_pt, digits=6))
    print("Matthews PT:")
    print(matthews_corrcoef(y_true_pt, y_predicted_pt))

    print("---- Spanish ----")
    y_true_es = []
    for pos, doc in docs_es.items():
        for extraction in doc["extractions"]:

            # if any(len(x) < 1 for x in extraction.values()):
            #     continue

            y_true_es.append(extraction["valid"])
    y_predicted_es = ["1"] * len(y_true_es)
    print(classification_report(y_true_es, y_predicted_es, digits=6))
    print("Matthews ES:")
    print(matthews_corrcoef(y_true_es, y_predicted_es))


def predict_udify(input_dict):
    import_submodules("udify")

    archive_dir = Path("models")

    config_file = archive_dir / "config.json"

    overrides = {}
    configs = [Params(overrides), Params.from_file(config_file)]
    params = util.merge_configs(configs)

    predictor = "udify_text_predictor"

    params["trainer"]["cuda_device"] = -1
    cuda_device = params["trainer"]["cuda_device"]

    archive = load_archive(archive_dir, cuda_device=cuda_device)

    predictor = Predictor.from_archive(archive, predictor)

    dataset_reader = predictor._dataset_reader
    tokenizer = dataset_reader.tokenizer

    for pos, item in input_dict.items():
        tokens = [word.text for word in tokenizer.split_words(item["phase"])]

        instance = dataset_reader.text_to_instance(tokens)

        item["prediction_udify"] = predictor.predict_instance(instance)
        print(".", end="")


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


def fill_vec_features(
        name,
        dependencies,
        feats,
        upos,
        possible_features,
        vec_features,
        fill_first_and_last,
):
    vec_features[f"{name}_size"] = len(upos)
    if fill_first_and_last:
        # UPOS
        for _upos in possible_features["upos"]:
            vec_features[f"{name}_first_upos_{_upos}"] = 0
            vec_features[f"{name}_last_upos_{_upos}"] = 0
        vec_features[f"{name}_first_upos_{upos[0]}"] = 1
        vec_features[f"{name}_last_upos_{upos[-1]}"] = 1

        # _dep
        for _dep in possible_features["dep"]:
            vec_features[f"{name}_first_dep_{_dep}"] = 0
            vec_features[f"{name}_last_dep_{_dep}"] = 0
        vec_features[f"{name}_first_dep_{dependencies[0]}"] = 1
        vec_features[f"{name}_last_dep_{dependencies[-1]}"] = 1

    for _dep in possible_features["dep"]:
        vec_features[f"{name}_dep_{_dep}"] = 0

    for _feat in possible_features["feats"]:
        vec_features[f"{name}_feat_{_feat}"] = 0

    for _upos in possible_features["upos"]:
        vec_features[f"{name}_upos_{_upos}"] = 0

    for _dep in dependencies:
        vec_features[f"{name}_dep_{_dep}"] += 1
    for _feat in feats:
        for single in _feat.split("|"):
            vec_features[f"{name}_feat_{single}"] += 1
    for _upos in upos:
        vec_features[f"{name}_upos_{_upos}"] += 1


def fill_dependencies_path(
        udify_prediction,
        possible_features,
        vec_features,
        arg1_start_pos,
        arg1_end_pos,
        rel_start_pos,
        rel_end_pos,
        arg2_start_pos,
        arg2_end_pos,
        arg1_dependencies,
        arg1_heads,
        rel_dependencies,
        rel_heads,
        arg2_dependencies,
        arg2_heads,
):
    def locate_pos(pos):
        if arg1_start_pos <= pos <= arg1_end_pos:
            return "arg1"
        elif rel_start_pos <= pos <= rel_end_pos:
            return "rel"
        elif arg2_start_pos <= pos <= arg2_end_pos:
            return "arg2"
        else:
            return "out"

    # Fill all possible combinations
    for _dep in possible_features["dep"]:
        for item_to in ["arg1", "rel", "arg2", "out"]:
            for item_from in ["arg1", "rel", "arg2"]:
                # if item_to == item_from:  # Let´s ignore a self reference
                #    continue
                vec_features[f"dep_{item_from}_to_{item_to}_{_dep}"] = 0

    # Arg 1 to rel
    # Arg 1 to Arg2
    # Rel to arg1
    # Rel to arg2
    # arg2 to rel
    # arg2 to arg1

    # TODO fix when head is 0 ( root)
    vec_features[f"dep_arg1_has_root"] = 0
    vec_features[f"dep_rel_has_root"] = 0
    vec_features[f"dep_arg2_has_root"] = 0

    for head in arg1_heads:
        head -= 1
        if head == -1:
            vec_features[f"dep_arg1_has_root"] = 1
            continue
        position = locate_pos(head)

        # if position == 'out' or position == 'arg1':
        #    continue
        tag = udify_prediction["predicted_dependencies"][head]
        vec_features[f"dep_arg1_to_{position}_{tag}"] += 1

    for head in rel_heads:
        head -= 1
        if head == -1:
            vec_features[f"dep_rel_has_root"] = 1
            continue
        position = locate_pos(head)
        # if position == 'out' or position == 'rel':
        #    continue
        tag = udify_prediction["predicted_dependencies"][head]
        vec_features[f"dep_rel_to_{position}_{tag}"] += 1

    for head in arg2_heads:
        head -= 1
        if head == -1:
            vec_features[f"dep_arg2_has_root"] = 1
            continue

        position = locate_pos(head)
        # if position == 'out' or position == 'arg2':
        #    continue
        tag = udify_prediction["predicted_dependencies"][head]
        vec_features[f"dep_arg2_to_{position}_{tag}"] += 1


def extract_features(extraction, udify_prediction, possible_features):
    arg1_start_pos, arg1_end_pos = extraction["arg1_pos"]
    rel_start_pos, rel_end_pos = extraction["rel_pos"]
    arg2_start_pos, arg2_end_pos = extraction["arg2_pos"]

    # Distance feature in percentage
    possible_features["arg1_to_rel_distance"] = (rel_start_pos - arg1_end_pos) / len(
        udify_prediction["words"]
    )
    possible_features["rel_to_arg2_distance"] = (arg2_start_pos - rel_end_pos) / len(
        udify_prediction["words"]
    )

    def get_feature(arg_start_pos, arg_end_pos):
        dependencies = udify_prediction["predicted_dependencies"][
                       arg_start_pos: arg_end_pos + 1
                       ]
        heads = udify_prediction["predicted_heads"][arg_start_pos: arg_end_pos + 1]
        feats = udify_prediction["feats"][arg_start_pos: arg_end_pos + 1]
        lemmas = udify_prediction["lemmas"][arg_start_pos: arg_end_pos + 1]
        upos = udify_prediction["upos"][arg_start_pos: arg_end_pos + 1]
        return dependencies, heads, feats, lemmas, upos

    final_features = {}

    arg1_dependencies, arg1_heads, arg1_feats, arg1_lemmas, arg1_upos = get_feature(
        arg1_start_pos, arg1_end_pos
    )

    fill_vec_features(
        "arg1",
        arg1_dependencies,
        arg1_feats,
        arg1_upos,
        possible_features,
        final_features,
        fill_first_and_last=True,
    )

    rel_dependencies, rel_heads, rel_feats, rel_lemmas, rel_upos = get_feature(
        rel_start_pos, rel_end_pos
    )
    fill_vec_features(
        "rel",
        rel_dependencies,
        rel_feats,
        rel_upos,
        possible_features,
        final_features,
        fill_first_and_last=True,
    )

    arg2_dependencies, arg2_heads, arg2_feats, arg2_lemmas, arg2_upos = get_feature(
        arg2_start_pos, arg2_end_pos
    )
    fill_vec_features(
        "arg2",
        arg2_dependencies,
        arg2_feats,
        arg2_upos,
        possible_features,
        final_features,
        fill_first_and_last=True,
    )

    # Create dependencies paths
    fill_dependencies_path(
        udify_prediction,
        possible_features,
        final_features,
        arg1_start_pos,
        arg1_end_pos,
        rel_start_pos,
        rel_end_pos,
        arg2_start_pos,
        arg2_end_pos,
        arg1_dependencies,
        arg1_heads,
        rel_dependencies,
        rel_heads,
        arg2_dependencies,
        arg2_heads,
    )

    return final_features


def get_transformed_dataset(input_dict, possibile_features, allowed_indexes=None):
    X = []
    y = []
    X_invalid = []
    y_invalid = []

    for pos, item in input_dict.items():

        if (allowed_indexes is not None) and (pos not in allowed_indexes):
            continue

        phrase_features = {}

        fill_vec_features(
            "phrase",
            item["prediction_udify"]["predicted_dependencies"],
            item["prediction_udify"]["feats"],
            item["prediction_udify"]["upos"],
            possibile_features,
            phrase_features,
            fill_first_and_last=False,
        )

        for extraction in item["extractions"]:

            if extraction["invalid_format"]:
                X_invalid.append(extraction)
                y_invalid.append(int(extraction["valid"]))
            else:
                extraction_features = extract_features(
                    extraction, item["prediction_udify"], possibile_features
                )
                combined_features = {**phrase_features, **extraction_features}
                X.append(combined_features)
                y.append(int(extraction["valid"]))

    return X, y, X_invalid, y_invalid


def set_extractions_features(input_dict, possibile_features):
    tokenizer = SpacyWordSplitter(language="xx_ent_wiki_sm")

    for pos, item in input_dict.items():
        tokens = item["prediction_udify"]["words"]
        print(f"Resultado para {item['phase']}")

        # Let get all possible features to make our dictionary
        for dep in item["prediction_udify"]["predicted_dependencies"]:
            possibile_features["dep"].add(dep)

        # possibile_features['first_last_dep'].add(dep)
        # possibile_features['first_last_dep'].add(item['prediction_udify']['predicted_dependencies'][0])

        for feat in item["prediction_udify"]["feats"]:
            for el in feat.split("|"):
                possibile_features["feats"].add(el)

        for upos in item["prediction_udify"]["upos"]:
            possibile_features["upos"].add(upos)

        # possibile_features['first_last_upos'].add(upos)
        # possibile_features['first_last_upos'].add(item['prediction_udify']['upos'][0])

        for extraction in item["extractions"]:

            arg1_tokens = [
                word.text for word in tokenizer.split_words(extraction["arg1"])
            ]
            arg1_start, arg1_end = find_sublist_match(tokens, arg1_tokens)
            if arg1_end == 0 and arg1_start > 0:
                extraction["invalid_format"] = True
                if extraction["valid"] == "1":
                    print(f"[ERRO] arg1 invalid {extraction['arg1']}")
                continue

            rel_tokens = [
                word.text for word in tokenizer.split_words(extraction["rel"])
            ]
            rel_start, rel_end = find_sublist_match(tokens, rel_tokens)

            if rel_end == 0 and rel_start > 0:
                extraction["invalid_format"] = True
                if extraction["valid"] == "1":
                    print(f"[ERRO] rel invalid {extraction['rel']}")
                continue

            arg2_tokens = [
                word.text for word in tokenizer.split_words(extraction["arg2"])
            ]
            arg2_start, arg2_end = find_sublist_match(tokens, arg2_tokens)

            if arg2_end == 0 and arg2_start > 0:
                extraction["invalid_format"] = True
                if extraction["valid"] == "1":
                    print(f"[ERRO] arg2 invalid {extraction['arg2']}")
                continue

            extraction["invalid_format"] = False
            extraction["arg1_pos"] = (arg1_start, arg1_end)
            extraction["rel_pos"] = (rel_start, rel_end)
            extraction["arg2_pos"] = (arg2_start, arg2_end)


def transformar_contracoes_pt(texto):
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

    return texto_convertido


def transformar_contracoes_es(texto):
    contracoes = {
        "al": "a el",
        "ofrecérselas": "ofrecér selas",
    }
    texto_convertido = texto
    for nome, replacement in contracoes.items():
        texto_convertido = re.sub(fr"\b{nome}\b", replacement, texto_convertido)

    return texto_convertido


def create_classifier(X_train, y_train, name="catboost"):
    train = []
    labels = []

    for X_item, y_item in zip(X_train, y_train):
        train.append([x for x in X_item.values()])
        labels.append(int(y_item))

    feature_names = list(X_train[0].keys())

    # TODO, use a interpretable

    if name == "catboost":
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=True,
        )
        model.fit(train, labels)

    elif name == "skope":
        model = DecisionListClassifier(max_depth=6, feature_names=feature_names)
        model.fit(train, labels)
        rules = model._extract_rules(model.sk_model_.rules_)
        print(rules)
    elif name == "explainable_boosting":
        model = ExplainableBoostingClassifier(n_jobs=8)
        model.fit(train, labels)
    elif name == "tabnet":
        model = TabNetClassifier()
        train = np.asarray(train, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        model.fit(train, labels, train, labels, max_epochs=250, patience=30)
    elif name == "sklearn_boost":
        model = HistGradientBoostingClassifier(
            max_iter=300, verbose=10
        )
        model.fit(train, labels)

    #

    # MLLP
    # structure = [len(train[0]), 2, 64, 32]
    # train = numpy.asarray(train, dtype=numpy.float32)
    # labels = numpy.asarray(labels, dtype=numpy.float32)
    # model = MLLP(structure, -1)
    # model.train(train, labels)

    # explain = model.explain_global()
    # print(explain)

    return model


def predict(model, X_predict, precision):
    train = []

    for X_item in X_predict:
        train.append([x for x in X_item.values()])

    train = np.asarray(train, dtype=np.float32)
    proba = model.predict_proba(train)

    final_prediction = []
    for x in proba:
        if x[0] > precision:
            final_prediction.append(0)
        else:
            final_prediction.append(1)

    return final_prediction

def generate_classification_report(model_lang,  X_train, y_train, X_predict,
                                   y_predict,
                                   y_predict_invalid):

    for model_name in ["catboost", "skope", "explainable_boosting", "tabnet", "sklearn_boost"]:

        if os.path.exists(f"{model_lang}_{model_name}.txt"):
            print(f"{model_lang}_{model_name}.txt exists, moving to next")
            continue

        gc.collect()
        model = create_classifier(X_train, y_train, model_name)
        gc.collect()
        with open(f"{model_lang}_{model_name}.txt", "w") as file_out:

            file_out.write(
                f"precision_at, accuracy, 1_precision, 1_recall, 1_f1-score, 1_support, 0_precision, 0_recall, 0_f1-score, 0_support, matthews\n")
            for precision_at in [0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.98, 0.99, 0.995, 0.999]:
                pred = predict(model, X_predict, precision_at)

                print(f"{model_lang}@{precision_at} - {model_name}")
                report = classification_report(y_predict + y_predict_invalid, pred + [0] * len(y_predict_invalid),
                                               output_dict=True)
                print(classification_report(y_predict + y_predict_invalid, pred + [0] * len(y_predict_invalid)))
                #print(classification_report([str(y) for y in true_y], predictions_ajusted))
                matthews = matthews_corrcoef(pred + y_predict_invalid, pred + [0] * len(y_predict_invalid))
                print(f"Matthews:{matthews}")


                file_out.write(
                    f"{precision_at},{report['accuracy']},{report['1']['precision']},"
                    f"{report['1']['recall']},{report['1']['f1-score']},{report['1']['support']},"
                    f"{report['0']['precision']},{report['0']['recall']},{report['0']['f1-score']},"
                    f"{report['0']['support']}, {matthews}\n")


def kfoldcv(indices, k=10, seed=42):
    size = len(indices)
    subset_size = round(size / k)
    random.Random(seed).shuffle(indices)
    subsets = [indices[x:x + subset_size] for x in range(0, len(indices), subset_size)]
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test:
                train.append(subset)
        kfolds.append((train, test))

    return kfolds


def evaluate(docs_pt, docs_en, docs_es):

    # Get folds
    if not os.path.exists('folds_english.pickle'):
        folds_english = kfoldcv([x for x in docs_en.keys()], k=5)
        with open('folds_english.pickle', 'wb') as handle:
            pickle.dump(folds_english, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('folds_english.pickle', 'rb') as handle:
            folds_english = pickle.load(handle)

    if not os.path.exists('folds_portuguese.pickle'):
        folds_portuguese = kfoldcv([x for x in docs_pt.keys()], k=5)
        with open('folds_portuguese.pickle', 'wb') as handle:
            pickle.dump(folds_portuguese, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('folds_portuguese.pickle', 'rb') as handle:
            folds_portuguese = pickle.load(handle)

    if not os.path.exists('folds_spanish.pickle'):
        folds_spanish = kfoldcv([x for x in docs_es.keys()], k=5)
        with open('folds_spanish.pickle', 'wb') as handle:
            pickle.dump(folds_spanish, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('folds_spanish.pickle', 'rb') as handle:
            folds_spanish = pickle.load(handle)







    X_pt, y_pt, X_pt_invalid, y_pt_invalid = get_transformed_dataset(docs_pt, possibile_features)
    X_en, y_en, X_en_invalid, y_en_invalid = get_transformed_dataset(docs_en, possibile_features)
    X_es, y_es, X_es_invalid, y_es_invalid = get_transformed_dataset(docs_es, possibile_features)


    # Zero-shot
    # pt+en -> es
    X_pt_en, y_pt_en = [], []
    X_pt_en.extend(X_pt)
    X_pt_en.extend(X_en)
    y_pt_en.extend(y_pt)
    y_pt_en.extend(y_en)

    generate_classification_report("zero_shot_pt+en_to_es", X_pt_en, y_pt_en, X_es,
                                   y_predict=y_es, y_predict_invalid=y_es_invalid)

    # pt+es -> en
    X_pt_es, y_pt_es = [], []
    X_pt_es.extend(X_pt)
    X_pt_es.extend(X_es)
    y_pt_es.extend(y_pt)
    y_pt_es.extend(y_es)

    # model_pt_es = create_classifier(X_pt_es, y_pt_es)
    # pred_en_by_pt_es = predict(model_pt_es, X_en)
    # print(classification_report(y_en, pred_en_by_pt_es))
    generate_classification_report("zero_shot_pt+es_to_en", X_pt_es, y_pt_es, X_en,
                                   y_predict=y_en, y_predict_invalid=y_en_invalid)

    # En+es -> pt
    X_en_es, y_en_es = [], []
    X_en_es.extend(X_en)
    X_en_es.extend(X_es)
    y_en_es.extend(y_en)
    y_en_es.extend(y_es)

    generate_classification_report("zero_shot_En+es_to_pt", X_en_es, y_en_es, X_pt,
                                   y_predict=y_pt, y_predict_invalid=y_pt_invalid)

    # Mono lingual and One SHOT
    for k in range(len(folds_english)):

        # Portuguese
        pt_train_ids = []
        for train_bucket in folds_portuguese[k][0]:
            pt_train_ids.extend(train_bucket)

        X_pt_train_fold, y_pt_train_fold, X_pt_train_fold_invalid, y_pt_train_fold_invalid = get_transformed_dataset(
            docs_pt, possibile_features, allowed_indexes=pt_train_ids)

        X_pt_test_fold, y_pt_test_fold, X_pt_test_fold_invalid, y_pt_test_fold_invalid = get_transformed_dataset(
            docs_pt, possibile_features, allowed_indexes=folds_portuguese[k][1])

        gc.collect()
        generate_classification_report(f"mono_pt_fold_{k}", X_pt_train_fold, y_pt_train_fold, X_pt_test_fold,
                                       y_predict=y_pt_test_fold, y_predict_invalid=y_pt_test_fold_invalid)

        # Spanish
        es_train_ids = []
        for train_bucket in folds_spanish[k][0]:
            es_train_ids.extend(train_bucket)

        X_es_train_fold, y_es_train_fold, X_es_train_fold_invalid, y_es_train_fold_invalid = get_transformed_dataset(
            docs_es, possibile_features, allowed_indexes=es_train_ids)

        X_es_test_fold, y_es_test_fold, X_es_test_fold_invalid, y_es_test_fold_invalid = get_transformed_dataset(
            docs_es, possibile_features, allowed_indexes=folds_spanish[k][1])

        gc.collect()
        generate_classification_report(f"mono_es_fold_{k}", X_es_train_fold, y_es_train_fold, X_es_test_fold,
                                       y_predict=y_es_test_fold, y_predict_invalid=y_es_test_fold_invalid)

        # English
        en_train_ids = []
        for train_bucket in folds_english[k][0]:
            en_train_ids.extend(train_bucket)

        X_en_train_fold, y_en_train_fold, X_en_train_fold_invalid, y_en_train_fold_invalid = get_transformed_dataset(
            docs_en, possibile_features, allowed_indexes=en_train_ids)

        X_en_test_fold, y_en_test_fold, X_en_test_fold_invalid, y_en_test_fold_invalid = get_transformed_dataset(
            docs_en, possibile_features, allowed_indexes=folds_english[k][1])

        gc.collect()
        generate_classification_report(f"mono_en_fold_{k}", X_en_train_fold, y_en_train_fold, X_en_test_fold,
                                       y_predict=y_en_test_fold, y_predict_invalid=y_en_test_fold_invalid)

        # ONE Shot FULL
        X_pt_en_oneshot = X_pt_en.copy()
        y_pt_en_oneshot = y_pt_en.copy()
        X_pt_en_oneshot.extend(X_es_train_fold)
        y_pt_en_oneshot.extend(y_es_train_fold)

        generate_classification_report(f"one_shot_full_pt+en_to_es_fold_{k}", X_pt_en_oneshot, y_pt_en_oneshot,
                                       X_es_test_fold,
                                       y_predict=y_es_test_fold, y_predict_invalid=y_es_test_fold_invalid)

        X_pt_es_oneshot = X_pt_es.copy()
        y_pt_es_oneshot = y_pt_es.copy()
        X_pt_es_oneshot.extend(X_en_train_fold)
        y_pt_es_oneshot.extend(y_en_train_fold)

        generate_classification_report(f"one_shot_full_pt+es_to_en_fold_{k}", X_pt_es_oneshot, y_pt_es_oneshot,
                                       X_en_test_fold,
                                       y_predict=y_en_test_fold, y_predict_invalid=y_en_test_fold_invalid)

        X_en_es_oneshot = X_en_es.copy()
        y_en_es_oneshot = y_en_es.copy()
        X_en_es_oneshot.extend(X_pt_train_fold)
        y_en_es_oneshot.extend(y_pt_train_fold)

        generate_classification_report(f"one_shot_full_En+es_to_pt_fold_{k}", X_en_es_oneshot, y_en_es_oneshot,
                                       X_pt_test_fold,
                                       y_predict=y_pt_test_fold, y_predict_invalid=y_pt_test_fold_invalid)

        # ONE Shot
        X_pt_en_oneshot = X_pt_en.copy()
        y_pt_en_oneshot = y_pt_en.copy()
        X_pt_en_oneshot.extend(X_es_test_fold)
        y_pt_en_oneshot.extend(y_es_test_fold)

        generate_classification_report(f"one_shot_pt+en_to_es_fold_{k}", X_pt_en_oneshot, y_pt_en_oneshot,
                                       X_es_train_fold,
                                       y_predict=y_es_train_fold, y_predict_invalid=y_es_train_fold_invalid)

        X_pt_es_oneshot = X_pt_es.copy()
        y_pt_es_oneshot = y_pt_es.copy()
        X_pt_es_oneshot.extend(X_en_test_fold)
        y_pt_es_oneshot.extend(y_en_test_fold)

        generate_classification_report(f"one_shot_pt+es_to_en_fold_{k}", X_pt_es_oneshot, y_pt_es_oneshot,
                                       X_en_train_fold,
                                       y_predict=y_en_train_fold, y_predict_invalid=y_en_train_fold_invalid)

        X_en_es_oneshot = X_en_es.copy()
        y_en_es_oneshot = y_en_es.copy()
        X_en_es_oneshot.extend(X_pt_test_fold)
        y_en_es_oneshot.extend(y_pt_test_fold)

        generate_classification_report(f"one_shot_En+es_to_pt_fold_{k}", X_en_es_oneshot, y_en_es_oneshot,
                                       X_pt_train_fold,
                                       y_predict=y_pt_train_fold, y_predict_invalid=y_pt_train_fold_invalid)


def average_folds():

    dict_results = {}
    for lang_name in ["pt+es_to_en", "pt+en_to_es", "En+es_to_pt"]:
        for model_name in ["catboost", "skope", "explainable_boosting", "tabnet", "sklearn_boost"]:
            key = f"one_shot_full_{lang_name}_{model_name}"
            dict_results[key] = dict()
            for k in range(5):
                file_name = f"one_shot_full_{lang_name}_fold_{k}_{model_name}.txt"
                with open(file_name, "r") as f_in:
                    lines = [x for x in f_in]
                    names = [x.strip() for x in lines[0].split(",")]
                    for line in lines[1:]:
                        segments = [x.strip() for x in line.split(",")]
                        if segments[0] not in dict_results[key]:
                            dict_results[key][segments[0]] = dict()
                        for pos, segment in enumerate(segments):
                            if pos == 0:
                                continue
                            internal_key = names[pos]
                            if internal_key not in dict_results[key][segments[0]]:
                                dict_results[key][segments[0]][internal_key] = list()
                            dict_results[key][segments[0]][internal_key].append(segment)

    for key, value in dict_results.items():
        with open(f"{key}.txt", "w") as f_out:
            f_out.write(", ".join(names) + "\n")
            for precision_at, values_precision in value.items():
                results_precision = [precision_at]
                for name in names[1:]:
                    if name in ['0_support', '1_support']:
                        segment_result = sum([int(x) for x in values_precision[name]])
                    else:
                        segment_result = sum([float(x) for x in values_precision[name]]) / len(values_precision[name])
                    results_precision.append(str(segment_result))
                results_precision.append("\n")
                f_out.write(",".join(results_precision))


if __name__ == "__main__":
    # average_folds()
    # exit(0)

    print("1 - Reading Dataset")
    # docs_en, docs_pt, docs_es = carregar_gamalho()
    docs_en, docs_pt, docs_es = carregar_lrec()

    print("1.1 - Dataset performance")
    report_performance(docs_en, docs_pt, docs_es)

    print("2.0 - Predict tags")

    print("2.1 - English")
    if not os.path.exists("processed_en.pickle"):
        predict_udify(docs_en)
        with open("processed_en.pickle", "wb") as handle:
            pickle.dump(docs_en, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(
            "2.1 - English SKIPPED - Delete processed_en.pickle if you want to process again"
        )
        with open("processed_en.pickle", "rb") as handle:
            docs_en = pickle.load(handle)

    print("2.2 - Portuguese")
    if not os.path.exists("processed_pt.pickle"):
        predict_udify(docs_pt)
        with open("processed_pt.pickle", "wb") as handle:
            pickle.dump(docs_pt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(
            "2.2 - Portuguese SKIPPED - Delete processed_pt.pickle if you want to process again"
        )
        with open("processed_pt.pickle", "rb") as handle:
            docs_pt = pickle.load(handle)

    print("2.3 - Espanhol")
    if not os.path.exists("processed_es.pickle"):
        predict_udify(docs_es)
        with open("processed_es.pickle", "wb") as handle:
            pickle.dump(docs_es, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(
            "2.3 - Espanhol SKIPPED - Delete processed_es.pickle if you want to process again"
        )
        with open("processed_es.pickle", "rb") as handle:
            docs_es = pickle.load(handle)

    print("3.0 - Prepare to classifier")
    possibile_features = {
        "dep": set(),
        # 'first_last_dep': set(),
        "feats": set(),
        "upos": set(),
        # 'first_last_upos': set()
    }
    set_extractions_features(docs_pt, possibile_features)
    set_extractions_features(docs_en, possibile_features)
    set_extractions_features(docs_es, possibile_features)
    print("4.0 - classifier")

    evaluate(docs_pt, docs_en, docs_es)
