import requests

from multioie.utils.dataset import read_dataset

tokens_list, tags_list = read_dataset("../../datasets/meu_dataset/test")
with open("saida_linguakit_gamalho.txt", encoding="utf-8", mode="w") as f_out:

    for tokens, tags in zip(tokens_list, tags_list):
        sentence = " ".join([token.token for token in tokens])

        params = {"lang": "pt", "text": sentence}

        f_out.write(f"SENT\t{sentence}\n")

        result = requests.post("https://api.linguakit.com/v2.0/rel", data=params)

        json_result = result.json()
        for result_line in json_result:
            f_out.write(result_line)
            f_out.write("\n")



# with open("../../datasets/meu_dataset/test/pud_100.txt", encoding="utf-8") as f_in:
#     with open("saida_linguakit.txt", encoding="utf-8", mode="w") as f_out:
#         for line in f_in:
#             params = {"lang": "pt", "text": line}
#
#             f_out.write(f"SENT\t{line}")
#
#             result = requests.post("https://api.linguakit.com/v2.0/rel", data=params)
#
#             json_result = result.json()
#             for result_line in json_result:
#                 f_out.write(result_line)
#                 f_out.write("\n")
