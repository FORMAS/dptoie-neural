
Configurando o Ambiente:

Instalar os pacotes atraves do arquivo de Pipenv:
pipenv install

Convertendo o conjunto de treino para o formato aceito pelo Allennlp, o formato CoNLL2012:

Rodar o script convert_to_conll.py


Para treinar:

Mudar os paths dos arquivos em openie_pt/config.json

Colocar na pasta nilc_embedding o GLOVE 100 dimensões(baixar em http://nilc.icmc.usp.br/embeddings)

Rodar o comando
allennlp train openie_pt/config.json -s saida_modelo

Para fazer o predict:
run_oie.py



Arquivos:
gamalho_dataset- O dataset original usado
meu_dataset - O dataset convertido para o formato conll2012
nilc_embedding - O embedding Glove
openie_pt - O arquivo de configuraçao utilizado no Allennlp
convert_to_conll.py - O script para converter o dataset para o conll2012
model_final.tar.gz - O modelo treinado em PT usando o nosso dataset
Pipfile - Os pacotes utilizados
run_oie.py- O Script para fazer a prediçao de novas frases
saida.gold_conll - O dataset em conll2012
saida.txt - As predicoes do nosso sistema
sentencas_teste.txt 0 As sentencas que nao foram utilizadas para treino


Referencias:
https://github.com/allenai/allennlp
https://github.com/gabrielStanovsky/supervised_oie_wrapper
