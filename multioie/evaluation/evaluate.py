import csv
import traceback
from pathlib import Path

from seqeval.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from multioie.evaluation.benchmark import Benchmark
from multioie.evaluation.matcher import Matcher
from multioie.model import AllenNLP
from multioie.model.openie_predictor import make_oie_string
from multioie.portuguese import PortugueseOIE
from multioie.utils.contractions import clean_extraction, transform_portuguese_contractions
from multioie.utils.convert_to_conll import save_data_as_conll, convert_to_conll
from multioie.utils.dataset import read_dataset




def evaluate_linguakit(true_tokens_list, true_tags_list, gold_dict):
    #file_output = Path(f"../../output_other_systems/linguakit/saida_linguakit.txt")
    file_output = Path(f"saida_linguakit_gamalho.txt")

    dataset = []
    with file_output.open(encoding="utf8") as f:
        actual_sent = None
        actual_extractions = []
        for line in f:
            parts = line.split("\t")
            if parts[0] == 'SENT' or len(parts) < 2:
                if actual_sent:

                    actual_sent = clean_extraction(actual_sent)
                    actual_sent = transform_portuguese_contractions(actual_sent.strip())
                    phrase_dict = {"phrase": actual_sent,
                                   "extractions": actual_extractions}

                    dataset.append(phrase_dict)
                if len(parts) < 2:
                    break
                actual_sent = parts[1]
                actual_extractions = []
            elif parts[0] == 'SENTID_1':

                arg1 = transform_portuguese_contractions(clean_extraction(parts[1].strip()))
                rel = transform_portuguese_contractions(clean_extraction(parts[2].strip()))
                arg2 = transform_portuguese_contractions(clean_extraction(parts[3].strip()))

                def clean_at_symbol(text):
                    text = text.replace("@", " ")
                    return text

                extraction_dict = {"arg1": clean_at_symbol(arg1),
                                   "rel": clean_at_symbol(rel),
                                   "arg2": clean_at_symbol(arg2),
                                   "confidence": 1
                                   }
                actual_extractions.append(extraction_dict)
    for pos_sentence, value in enumerate(dataset):
        convert_to_conll(value, pos_sentence)

    folder_output = Path(f"../../output_other_systems/linguakit/")

    conll_output_path = folder_output / "saida_conll_linguakit.conll"
    save_data_as_conll(0, conll_output_path, dataset)

    linguakit_tokens_list, linguakit_tags_list = read_dataset(folder_output)

    pred_tags_for_model = []
    true_tag_list = []

    file_out = open(folder_output / f"linguakit_result.csv", "w")
    file_out.write(
        f"model_name, accuracy, precision, recall, f1-score\n"
    )

    for result, true_tags in zip(linguakit_tags_list, true_tags_list):

        tags = sorted(result)

        print(f"I found {len(tags)=} for {len(true_tags)=}")
        for pos, individual in enumerate(sorted(true_tags)):
            if pos >= len(tags):
                pred_tags_for_model.append(["O"] * len(individual))
            else:
                pred_tags_for_model.append(tags[pos])

            true_tag_list.append(individual)

    b = Benchmark()

    # Transform in dictionary
    predict_dict = dict()
    for line in dataset:
        predict_dict[line["phrase"]] = line["extractions"]

    b.compare(gold=gold_dict, predicted=predict_dict, matchingFunc=Matcher.identicalMatch, output_fn="curve_linguakit.txt")


    # report = classification_report(true_tag_list, pred_tags_for_model)
    # print(report)
    #
    # accuracy = accuracy_score(true_tag_list, pred_tags_for_model)
    # precision = precision_score(true_tag_list, pred_tags_for_model)
    # recall = recall_score(true_tag_list, pred_tags_for_model)
    # f1 = f1_score(true_tag_list, pred_tags_for_model)
    #
    # file_out.write(
    #     f"linguakit,{accuracy},{precision},{recall},{f1}\n"
    # )


def evaluate_dpt(true_tokens_list, true_tags_list, gold_dict):
    file_output = Path(f"../../output_other_systems/dptoie/extractedFactsByDpOIE_gamalho.csv")

    dataset = []
    with file_output.open(encoding="utf8") as f:
        actual_sent = None
        actual_extractions = []
        csv_reader = csv.DictReader(f, delimiter=';')

        for row in csv_reader:

            sentenca = row["SENTENÇA"]
            if len(sentenca) > 0:
                if actual_sent:

                    actual_sent = clean_extraction(actual_sent)
                    actual_sent = transform_portuguese_contractions(actual_sent.strip())
                    phrase_dict = {"phrase": actual_sent,
                                   "extractions": actual_extractions}

                    dataset.append(phrase_dict)

                actual_sent = sentenca
                actual_extractions = []
            else:

                arg1 = transform_portuguese_contractions(clean_extraction(row["ARG1"].strip()))
                rel = transform_portuguese_contractions(clean_extraction(row["REL"].strip()))
                arg2 = transform_portuguese_contractions(clean_extraction(row["ARG2"].strip()))

                def clean_at_symbol(text):
                    #text = text.replace("@", " ")
                    return text

                extraction_dict = {"arg1": clean_at_symbol(arg1),
                                   "rel": clean_at_symbol(rel),
                                   "arg2": clean_at_symbol(arg2),
                                   "confidence": 1
                                   }
                actual_extractions.append(extraction_dict)
    for pos_sentence, value in enumerate(dataset):
        convert_to_conll(value, pos_sentence)

    folder_output = Path(f"../../output_other_systems/dptoie/")

    conll_output_path = folder_output / "saida_conll_dptoie_gamalho.conll"
    save_data_as_conll(0, conll_output_path, dataset)

    dptoie_tokens_list, dptoie_tags_list = read_dataset(folder_output)

    pred_tags_for_model = []
    true_tag_list = []

    file_out = open(folder_output / f"dptoie_result.csv", "w")
    file_out.write(
        f"model_name, accuracy, precision, recall, f1-score\n"
    )

    for result, true_tags in zip(dptoie_tags_list, true_tags_list):

        tags = sorted(result)

        print(f"I found {len(tags)=} for {len(true_tags)=}")

        # TODO, this eval is WRONG!
        for pos, individual in enumerate(sorted(true_tags)):
            if pos >= len(tags):
                pred_tags_for_model.append(["O"] * len(individual))
            else:
                pred_tags_for_model.append(tags[pos])

            true_tag_list.append(individual)

    b = Benchmark()

    # Transform in dictionary
    predict_dict = dict()
    for line in dataset:
        predict_dict[line["phrase"]] = line["extractions"]

    b.compare(gold=gold_dict, predicted=predict_dict, matchingFunc=Matcher.identicalMatch,  output_fn="curve_dpt.txt")

    #
    # accuracy = accuracy_score(true_tag_list, pred_tags_for_model)
    # precision = precision_score(true_tag_list, pred_tags_for_model)
    # recall = recall_score(true_tag_list, pred_tags_for_model)
    # f1 = f1_score(true_tag_list, pred_tags_for_model)
    #
    # file_out.write(
    #     f"dptoie,{accuracy},{precision},{recall},{f1}\n"
    # )


def hyperparameter_search(tokens_list, tags_list, gold_dict):
    models_path = Path(f"../../saida_novo/").resolve()
    #folders = models_path.glob('*')
    folders = [models_path]

    file_out = open(f"hyperparameter_search_gamalho.csv", "w")
    file_out.write(
        f"model_name, auc, precision, recall, f1-score\n"
    )

    for model_folder in folders:

        model_name = model_folder.parts[-1]

        #if model_name not in ["both", "features", "variations", "none"]:
        #    continue
        #if "LearningType.LSTM_384_3_EmbeddingType.BERT_PT_OptimizerType.MADGRAD" not in model_name:
        #    continue

        # Code for ablation

        # if model_name == "both":
        #     AllenNLP.DISABLE_RICH_FEATURES = True
        #     AllenNLP.DISABLE_VARIATION_GENERATOR = True
        # elif model_name == "features":
        #     AllenNLP.DISABLE_RICH_FEATURES = True
        #     AllenNLP.DISABLE_VARIATION_GENERATOR = False
        # elif model_name == "variations":
        #     AllenNLP.DISABLE_RICH_FEATURES = False
        #     AllenNLP.DISABLE_VARIATION_GENERATOR = True
        # elif model_name == "none":
        #     AllenNLP.DISABLE_RICH_FEATURES = False
        #     AllenNLP.DISABLE_VARIATION_GENERATOR = False

        try:

            pred_dict, pred_tags_for_model, true_tag_list = process_using_model(model_folder, tags_list, tokens_list)

            #

            b = Benchmark()

            # Transform in dictionary
            precision_recall_f1, auc = b.compare(gold=gold_dict, predicted=pred_dict, matchingFunc=Matcher.identicalMatch,
                      output_fn=f"curve_{model_folder.parts[-1]}.txt")

            report = classification_report(true_tag_list, pred_tags_for_model)
        except:
            traceback.print_exc()
            print(f"Erro processando {model_name=}")
            continue

        # accuracy = accuracy_score(true_tag_list, pred_tags_for_model)
        # precision = precision_score(true_tag_list, pred_tags_for_model)
        # recall = recall_score(true_tag_list, pred_tags_for_model)
        # f1 = f1_score(true_tag_list, pred_tags_for_model)

        accuracy = auc
        precision, recall, f1 = precision_recall_f1

        file_out.write(
            f"{model_name},{accuracy},{precision},{recall},{f1}\n"
        )
        file_out.flush()

        print(report)

    file_out.close()


def process_using_model(model_folder, tags_list, tokens_list):
    oie_system = PortugueseOIE(model_folder)
    print(f"Using model {model_folder}")

    pred_tags_for_model = []
    true_tag_list = []
    pred_dict = dict()
    for tokens, true_tags in zip(tokens_list, tags_list):
        try:
            result = oie_system.predict(tokens)
        except:
            print(f"Erro processando sentenca")
            continue

        tags = []
        for verb in result['verbs']:
            tags.append(verb['tags'])

        sentence = " ".join([x.token for x in tokens])
        pred_dict[sentence] = []

        for extraction in tags:
            dict_extraction = dict()
            arg1, rel, arg2 = get_pieces_from_tagged(tokens, extraction)
            dict_extraction["arg1"] = arg1
            dict_extraction["rel"] = rel
            dict_extraction["arg2"] = arg2
            dict_extraction["confidence"] = 1
            pred_dict[sentence].append(dict_extraction)

        tags = sorted(tags)

        print(f"I found {len(tags)=} for {len(true_tags)=}")
        for pos, individual in enumerate(sorted(true_tags)):
            if pos >= len(tags):
                pred_tags_for_model.append(["O"] * len(individual))
            else:
                pred_tags_for_model.append(tags[pos])

            true_tag_list.append(individual)
    return pred_dict, pred_tags_for_model, true_tag_list


def get_pieces_from_tagged(tokens, tags) -> str:
    """
    Converts a list of model outputs (i.e., a list of lists of bio tags, each
    pertaining to a single word), returns an inline bracket representation of
    the prediction.
    """
    frame = []
    chunk = []
    words = [token.token for token in tokens]

    rel = []
    arg1 = []
    arg2 = []

    processing_tag = None
    for (token, tag) in zip(words, tags):
        if tag.startswith("I-"):
            chunk.append(token)
        else:
            if chunk:
                #frame.append("[" + " ".join(chunk) + "]")

                if processing_tag == "ARG0":
                    arg1 = chunk
                elif processing_tag == "ARG1":
                    arg2 = chunk
                elif processing_tag == "V":
                    rel = chunk
                else:
                    raise ValueError(f"Unknown processing tag: {processing_tag}" )

                chunk = []

            if tag.startswith("B-"):
                processing_tag = tag[2:]
                chunk.append(token)
            elif tag == "O":
                frame.append(token)

    if chunk:
        if processing_tag == "ARG0":
            arg1 = chunk
        elif processing_tag == "ARG1":
            arg2 = chunk
        elif processing_tag == "V":
            rel = chunk
        else:
            raise ValueError(f"Unknown processing tag: {processing_tag}")

    return " ".join(arg1), " ".join(rel), " ".join(arg2)


def main():

    tokens_list, tags_list = read_dataset("../../datasets/meu_dataset/test")

    gold_dict = dict()
    for tokens, tags in zip(tokens_list, tags_list):
        sentence = " ".join([token.token for token in tokens])
        gold_dict[sentence] = []
        for extraction in tags:

            dict_extraction = dict()
            arg1, rel, arg2 = get_pieces_from_tagged(tokens, extraction)
            dict_extraction["arg1"] = arg1
            dict_extraction["rel"] = rel
            dict_extraction["arg2"] = arg2
            gold_dict[sentence].append(dict_extraction)


    #evaluate_linguakit(tokens_list, tags_list, gold_dict)

    #evaluate_dpt(tokens_list, tags_list, gold_dict)

    hyperparameter_search(tokens_list, tags_list, gold_dict)

    exit(0)



if __name__ == "__main__":
    main()
