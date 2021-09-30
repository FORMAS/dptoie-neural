import re
from pathlib import Path


def carregar_coling():
    # Portuguese dataset
    dataset_pt = dict()
    for name in ["coling2020.txt", "200-sentences-pt-PUD.txt"]:
        pt = Path(f"../datasets/coling_PUD/{name}")
        with open(pt, "r", encoding="utf-8") as f_pt:
            actual_pos = None
            for line in f_pt:
                line = line.strip()
                pos, phase = line.split("\t", 1)

                if pos.isnumeric() and phase.count("\t") < 1:
                    actual_pos = int(pos)
                    phrase = transformar_contracoes_pt(phase.strip()).strip(",.")
                    phrase = re.sub(r',|\.|"', "", phrase)
                    dataset_pt[actual_pos] = {"phase": phrase, "extractions": []}
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

    return dataset_pt


def carregar_pud200():
    # Portuguese dataset
    dataset_pt = dict()
    for name in ["200-sentences-pt-PUD.txt"]:
        pt = Path(f"../datasets/coling_PUD/{name}")
        with open(pt, "r", encoding="utf-8") as f_pt:
            actual_pos = None
            for line in f_pt:
                line = line.strip()
                pos, phase = line.split("\t", 1)

                if pos.isnumeric() and phase.count("\t") < 1:
                    actual_pos = int(pos)
                    phrase = transformar_contracoes_pt(phase.strip()).strip(",.")
                    phrase = re.sub(r',|\.|"', "", phrase)
                    dataset_pt[actual_pos] = {"phase": phrase, "extractions": []}
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

    return dataset_pt


def carregar_pud100_final():
    # Portuguese dataset
    dataset_pt = dict()
    for name in ["coling2020.txt"]:
        pt = Path(f"datasets/coling_PUD/{name}")
        with open(pt, "r", encoding="utf-8") as f_pt:
            actual_pos = None
            for line in f_pt:
                line = line.strip()
                pos, phase = line.split("\t", 1)

                if pos.isnumeric() and phase.count("\t") < 1:
                    actual_pos = int(pos)
                    phrase = transformar_contracoes_pt(phase.strip()).strip(",.")
                    phrase = re.sub(r',|\.|"', "", phrase)
                    dataset_pt[actual_pos] = {"phase": phrase, "extractions": []}
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

    return dataset_pt


def carregar_gamalho():
    # # English dataset
    dataset_en = dict()
    #
    # en = Path("Dataset/gamalho/en/sentences.txt")
    # with open(en, 'r', encoding='utf-8') as f_en:
    #     for line in f_en:
    #         pos, phase = line.split('\t')
    #         dataset_en[int(pos)] = {"phase": phase,
    #                                 "extractions": []
    #                                 }
    #
    # en = Path("Dataset/gamalho/en/extractions-all-labeled.txt")
    # with open(en, 'r', encoding='utf-8') as f_en:
    #     for line in f_en:
    #         if '\t' in line:
    #             partes = line.split("\t")
    #             pos = int(partes[0])
    #             arg1 = partes[1].strip('"')
    #             rel = partes[2].strip('"')
    #             arg2 = partes[3].strip('"')
    #             valid = partes[-1]
    #
    #             dataset_en[pos]['extractions'].append({"arg1": arg1,
    #                                                    "rel": rel,
    #                                                    "arg2": arg2,
    #                                                    "valid": valid.strip()})

    # Portuguese dataset
    dataset_pt = dict()
    dataset_es = dict()

    pt = Path("Dataset/gamalho/pt/sentences.txt")
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            line = line.strip()
            pos, phase = line.split("\t", 1)
            dataset_pt[int(pos)] = {
                "phase": transformar_contracoes_pt(phase.strip()),
                "extractions": [],
            }

    pt = Path("Dataset/gamalho/pt/argoe-pt-labeled.csv")
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            if "\t" in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = partes[1].strip('"')
                rel = partes[2].strip('"')
                arg2 = partes[3].strip('"')
                valid = partes[-1]

                dataset_pt[pos]["extractions"].append(
                    {
                        "arg1": transformar_contracoes_pt(arg1),
                        "rel": transformar_contracoes_pt(rel),
                        "arg2": transformar_contracoes_pt(arg2),
                        "valid": valid.strip(),
                    }
                )

    return dataset_pt


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
                dataset_pt[actual_pos] = {"phase": partes[1].strip(), "extractions": []}
            else:
                partes = line.split("\t")
                arg1 = partes[0].strip('"')
                rel = partes[1].strip('"')
                arg2 = partes[2].strip('"')
                valid = partes[-2]

                dataset_pt[actual_pos]["extractions"].append(
                    {"arg1": arg1, "rel": rel, "arg2": arg2, "valid": valid.strip()}
                )

    return dataset_pt


def report_performance(docs_pt):
    from sklearn.metrics import classification_report, matthews_corrcoef

    print("---- Portuguese ----")
    y_true_pt = []
    for pos, doc in docs_pt.items():
        for extraction in doc["extractions"]:

            if any(len(x) < 1 for x in extraction.values()):
                continue

            y_true_pt.append(extraction["valid"])
    y_predicted_pt = ["1"] * len(y_true_pt)
    print(classification_report(y_true_pt, y_predicted_pt, digits=6))
    print("Matthews PT:")
    print(matthews_corrcoef(y_true_pt, y_predicted_pt))


if __name__ == "__main__":
    print("1 - Reading Dataset")
    docs_pt = carregar_pud200()
    docs_coling = carregar_coling()
    report_performance(docs_coling)

    print("1.1 - Dataset performance")
    docs_coling = carregar_coling()
    # report_performance(docs_coling)
