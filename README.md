# DPTOIE-Neural

## Configurando o Ambiente

Instalar os pacotes atraves do arquivo de Pipenv:
````
pipenv install
````
Rodar o script
```
convert_to_conll.py
```
Convertendo o conjunto de treino para o formato aceito pelo Allennlp: o formato CoNLL2012.

## Para treinar:

### Atualizar os paths dos arquivos em 
```
openie_pt/config.json
```

Colocar na pasta nilc_embedding o GLOVE 100 de dimensões disponível no [site do NILC](http://nilc.icmc.usp.br/embeddings).

### Treinar o modelo
````
allennlp train openie_pt/config.json -s saida_modelo
````
### Para fazer o predict:
````
run_oie.py
````
### Arquivos:
* **gamalho_dataset** - O dataset original usado
* **meu_dataset** - O dataset convertido para o formato conll2012
* **nilc_embedding** - O embedding Glove
* **openie_pt** - O arquivo de configuraçao utilizado no Allennlp
* **convert_to_conll.py** - O script para converter o dataset para o conll2012
* **model_final.tar.gz** - O modelo treinado em PT usando o nosso dataset
* **Pipfile** - Os pacotes utilizados
* **run_oie.py** - O Script para fazer a prediçao de novas frases
* **saida.gold_conll** - O dataset em conll2012
* **saida.txt** - As predicoes do nosso sistema
* **sentencas_teste.txt** - As sentencas que nao foram utilizadas para treino
* **csv_to_labeled.py** - O script que converte o corpus avaliation_pragmOIE_FULL no formato CSV ceten200 e wiki200 aceito pelo script **convert_to_conll.py**


### Referencias:
* https://github.com/allenai/allennlp
* https://github.com/gabrielStanovsky/supervised_oie_wrapper
