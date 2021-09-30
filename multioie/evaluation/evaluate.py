import csv
from pathlib import Path

from seqeval.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from multioie.evaluation.benchmark import Benchmark
from multioie.evaluation.matcher import Matcher
from multioie.model.openie_predictor import make_oie_string
from multioie.portuguese import PortugueseOIE
from multioie.utils.contractions import clean_extraction, transform_portuguese_contractions
from multioie.utils.convert_to_conll import save_data_as_conll, convert_to_conll
from multioie.utils.dataset import read_dataset


def generate_classification_report(model_lang, model_name, predictions, true_y):
    ir = IsotonicRegression()
    # ir.fit(predictions, true_y)
    # results_en = ir.predict(predictions)

    with open(f"{model_lang}_{model_name}.txt", "a") as file_out:
        file_out.write(
            f"precision_at, accuracy, 1_precision, 1_recall, 1_f1-score, 1_support, 0_precision, 0_recall, 0_f1-score, 0_support, matthews\n"
        )
        for precision_at in [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.98,
            0.99,
            0.995,
            0.999,
            0.9999,
        ]:
            predictions_ajusted = [
                x["label"] if x["confidence"] >= precision_at else "1" for x in predictions
            ]

            print(f"{model_lang}@{precision_at} - {model_name}")
            report = classification_report(
                [str(y) for y in true_y],
                [str(y_a) for y_a in predictions_ajusted],
                output_dict=True,
            )
            # print(classification_report([str(y) for y in true_y], predictions_ajusted))
            print("Matthews:")
            matthews = matthews_corrcoef(
                [str(y) for y in true_y], [str(y_a) for y_a in predictions_ajusted]
            )

            file_out.write(
                f"{precision_at},{report['accuracy']},{report['1']['precision']},{report['1']['recall']},{report['1']['f1-score']},{report['1']['support']},{report['0']['precision']},{report['0']['recall']},{report['0']['f1-score']},{report['0']['support']}, {matthews}\n"
            )


def evaluate_portnoie():
    pass


def evaluate_graph(docs_pt):
    k = 5
    folds = kfoldcv([x for x in docs_pt.keys()], k=k)

    # for model_type in [CNN_Model, CNN_GRU_Model]:
    for model_type in [CNN_Model]:

        # Vamos fazer o K-Fold agora
        total_y = [[], []]
        total_y_score = [[], []]

        for k in range(len(folds)):

            train_ids = []
            for train_bucket in folds[k][0]:
                train_ids.extend(train_bucket)

            x_train, y_train = extractions_to_flat(docs_pt, indexes=train_ids)
            x_test, y_test = extractions_to_flat(docs_pt, indexes=folds[k][1])

            model = create_model(x_train, y_train, model_type)

            y_en_pred_top_k = model.predict_top_k_class(x_test, top_k=2)
            m_type = "sentence" if DO_SENTENCE_EMBEDDING else "regular"
            model_str = str(model_type).split(".")[-1].split("'")[0]
            name = f"{model_str}_{m_type}_{len(docs_pt)}"

            for label in [0, 1]:
                y_score = []
                y_test_pred = []

                for x in y_en_pred_top_k:
                    if x["label"] == label:
                        y_score.append(x["confidence"])
                        y_test_pred.append(1)
                    else:
                        y_score.append(1.0 - x["confidence"])
                        y_test_pred.append(0)

                total_y_score[label].extend(y_score)
                total_y[label].extend(y_test)

            generate_classification_report(f"fold_{k}", name, y_en_pred_top_k, y_test)

        # Gerar o grafico
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])

        bc = BinaryClassification(total_y[1], total_y_score[1], labels=["Class 1", "Class 2"])
        plt.figure(figsize=(15, 10))
        plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
        bc.plot_roc_curve()
        plt.savefig(f"{name}.png")

        average_folds(name, k=k)


def evaluate_linguakit(true_tokens_list, true_tags_list, gold_dict):
    file_output = Path(f"../../output_other_systems/linguakit/saida_linguakit.txt")

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
    file_output = Path(f"../../output_other_systems/dptoie/extractedFactsByDpOIE.csv")

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

    conll_output_path = folder_output / "saida_conll_dptoie.conll"
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
    models_path = Path(f"../../models/").resolve()
    folders = models_path.glob('*')

    file_out = open(f"hyperparameter_search.csv", "w")
    file_out.write(
        f"model_name, accuracy, precision, recall, f1-score\n"
    )

    for model_folder in folders:
        oie_system = PortugueseOIE(model_folder)
        print(f"Using model {model_folder}")

        pred_tags_for_model = []
        true_tag_list = []

        pred_dict = dict()

        for tokens, true_tags in zip(tokens_list, tags_list):
            result = oie_system.predict(tokens)
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

            # if len(true_tag_list) > 10:
            #    break

        #

        b = Benchmark()

        # Transform in dictionary
        b.compare(gold=gold_dict, predicted=pred_dict, matchingFunc=Matcher.identicalMatch,
                  output_fn=f"curve_{model_folder.parts[-1]}.txt")

        report = classification_report(true_tag_list, pred_tags_for_model)

        accuracy = accuracy_score(true_tag_list, pred_tags_for_model)
        precision = precision_score(true_tag_list, pred_tags_for_model)
        recall = recall_score(true_tag_list, pred_tags_for_model)
        f1 = f1_score(true_tag_list, pred_tags_for_model)

        file_out.write(
            f"{model_folder.parts[-1]},{accuracy},{precision},{recall},{f1}\n"
        )

        print(report)

    file_out.close()


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
