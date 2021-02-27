from pathlib import Path
import csv

i=0

pt = Path("pragmatic_dataset/wiki200.csv")
file = 'pragmatic_dataset/wiki200-labeled.csv'
sents = 'pragmatic_dataset/wiki200.txt'

with open(file, 'w', encoding='utf-8', newline='') as f_out:
    with open(pt, 'r', encoding='utf-8') as f_pt:
        for line in f_pt:
            if i > 0:
                partes = line.split('\"')
                if not partes[1] == '':
                    id          = partes[1].replace("\"","")
                sentence    = partes[3].replace("\"","")
                arg1        = partes[5].replace("\"","")
                rel         = partes[7].replace("\"","")
                arg2        = partes[9].replace("\"","")
                anotation   = partes[13].replace("\"","")
                writer = csv.writer(f_out, delimiter='\t')
                if anotation != '0' and anotation != '1':
                    continue
                writer.writerow([id.strip(), arg1.strip(), rel.strip(), arg2.strip(), anotation.strip()])
            i += 1
print('Finished - Convert csv to labeled! \nWrited in ', file)
i=0
with open(sents, 'w', encoding='utf-8') as s_out:
    with open(pt, 'r', encoding='utf-8') as f_pt:
        for line in f_pt:
            if i > 0:
                partes = line.split('\"')
                if not partes[1] == '':
                    id          = partes[1].replace("\"","")
                    sentence    = partes[3].replace("\"","")
                    if len(sentence) > 0:
                        if sentence[-1] == '.':
                            sentence = sentence.replace(sentence[-1], ' .')
                        s_out.write(id.strip() + '\t' + sentence.strip() + '\n')
            i += 1
print('Finished - Create file\'s txt with sentences from CSV ! \nWrited in ', sents)