from pathlib import Path


def load_dataset():
    dataset_pt = dict()

    pt = Path("../gamalho_dataset/sentences.txt")
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            line = line.strip()
            pos, phase = line.split("\t", 1)
            dataset_pt[int(pos)] = {"phase": phase.strip(), "extractions": []}

    pt = Path("../gamalho_dataset/argoe-pt-labeled.csv")
    with open(pt, "r", encoding="utf-8") as f_pt:
        for line in f_pt:
            if "\t" in line:
                partes = line.split("\t")
                pos = int(partes[0])
                arg1 = partes[1].strip('"')
                rel = partes[2].strip('"')
                arg2 = partes[3].strip('"')
                valid = partes[-1].strip()

                if valid != "1":
                    continue  # so queremos pegar as positivas

                dataset_pt[pos]["extractions"].append(
                    {"arg1": arg1, "rel": rel, "arg2": arg2, "valid": int(valid)}
                )

    return dataset_pt


if __name__ == "__main__":
    dataset = load_dataset()
