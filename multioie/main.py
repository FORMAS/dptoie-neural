import traceback
from pathlib import Path
import random

from multioie.model.AllenNLP import (
    EmbeddingType,
    LearningType,
    OptimizerType,
)
import typer

from multioie.portuguese import PortugueseOIE

app = typer.Typer()


@app.command()
def train(
    input_path_str: str, model_output_path_str: Path, max_iterations: int = 30, layers: int = 2
):
    input_path = Path(input_path_str).resolve()
    output_path = Path(model_output_path_str).resolve()

    oie_system = PortugueseOIE()
    oie_system.train(
        input_path=input_path,
        destination_model_path=output_path,
        max_iterations=max_iterations,
        layers=layers,
        embedding=EmbeddingType.SELF_100,
        network=LearningType.LSTM,
    )


def get_options():
    network_options = [LearningType.LSTM, LearningType.SRUPP] # LearningType.XTRANSFORMER,
    hidden_dimension_options = [384, 512]
    layers_options = [2, 3]
    embedding_options = [
        #EmbeddingType.GLOVE,
        #EmbeddingType.FLAIR_DIARIOS_BI_1024,
        EmbeddingType.BERT_PT
        # EmbeddingType.FLAIR_DIARIOS_BI_2048,
    ]
    optimizer_options = [
        # OptimizerType.SGD,
        #OptimizerType.RADAM,
        OptimizerType.RANGER
    ]

    options = []
    for network in network_options:
        for hidden_dimension in hidden_dimension_options:
            for embedding in embedding_options:
                for layers in layers_options:
                    for optimizer in optimizer_options:
                        option = {
                            "network": network,
                            "hidden_dimension": hidden_dimension,
                            "layers": layers,
                            "embedding": embedding,
                            "optimizer": optimizer,
                        }
                        options.append(option)

    return options


@app.command()
def create_multiple_models():
    input_path = Path("../datasets/meu_dataset/").resolve()
    options = get_options()
    random.shuffle(options)

    for option in options:
        str_key = "_".join([str(x) for x in option.values()])
        output_path = Path(f"../models/{str_key}")

        if output_path.exists():
            print(f"Skipping existing model: {str_key}")
            continue
        print(f"Processing {str_key}")
        try:
            output_path.mkdir(parents=True)

            oie_system = PortugueseOIE()
            print(f"TRAINING WITH {str_key}")
            oie_system.train(
                input_path=input_path,
                destination_model_path=output_path,
                max_iterations=100,
                **option,
            )
        except:
            traceback.print_exc()


@app.command()
def predict(input_str: str, model_path: Path):
    oie_system = PortugueseOIE(model_path=model_path)
    result = oie_system.predict_str(input_str)
    print(result)


if __name__ == "__main__":
    app()
