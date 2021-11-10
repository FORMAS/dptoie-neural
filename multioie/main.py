import traceback
from pathlib import Path
import random

from multioie.model import AllenNLP
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
    max_iterations = 70
    layers = 3
    oie_system.train(
        input_path=input_path,
        destination_model_path=output_path,
        max_iterations=max_iterations,
        layers=layers,
        embedding=EmbeddingType.BERT_PT,
        network=LearningType.LSTM,
        optimizer=OptimizerType.MADGRAD,
        hidden_dimension=384,
        batch_size=32
    )


def get_options():
    network_options = [LearningType.LSTM, LearningType.SRUPP] # LearningType.XTRANSFORMER,
    hidden_dimension_options = [384, 512, 768]
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
        #OptimizerType.RANGER,
        OptimizerType.MADGRAD
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
def create_ablation_models():
    input_path = Path("../datasets/meu_dataset/").resolve()

    for option in ["both", "features", "variations", "none"]:
        str_key = option
        output_path = Path(f"../models/{str_key}")

        output_path_best = Path(f"../models/{str_key}") / "model_final" / "best.th"
        if output_path_best.exists():
            print(f"Skipping existing model: {str_key}")
            continue

        if option == "both":
            AllenNLP.DISABLE_RICH_FEATURES = True
            AllenNLP.DISABLE_VARIATION_GENERATOR = True
        elif option == "features":
            AllenNLP.DISABLE_RICH_FEATURES = True
            AllenNLP.DISABLE_VARIATION_GENERATOR = False
        elif option == "variations":
            AllenNLP.DISABLE_RICH_FEATURES = False
            AllenNLP.DISABLE_VARIATION_GENERATOR = True
        elif option == "none":
            AllenNLP.DISABLE_RICH_FEATURES = False
            AllenNLP.DISABLE_VARIATION_GENERATOR = False


        print(f"Processing {str_key}")
        output_path.mkdir(parents=True, exist_ok=True)

        oie_system = PortugueseOIE()
        print(f"TRAINING WITH {str_key}")
        oie_system.train(
            input_path=input_path,
            destination_model_path=output_path,
            max_iterations=80,
            batch_size=32,
            layers = 3,
            network = LearningType.LSTM,
            embedding = EmbeddingType.BERT_PT,
            optimizer = OptimizerType.MADGRAD,
            hidden_dimension = 384,
        )



@app.command()
def create_multiple_models():
    input_path = Path("../datasets/meu_dataset/").resolve()
    options = get_options()
    random.shuffle(options)

    for option in options:
        str_key = "_".join([str(x) for x in option.values()])
        output_path = Path(f"../models/{str_key}")

        output_path_best = Path(f"../models/{str_key}") / "model_final" / "best.th"

        if output_path_best.exists():
            print(f"Skipping existing model: {str_key}")
            continue
        print(f"Processing {str_key}")
        try:
            output_path.mkdir(parents=True, exist_ok=True)

            oie_system = PortugueseOIE()
            print(f"TRAINING WITH {str_key}")
            oie_system.train(
                input_path=input_path,
                destination_model_path=output_path,
                max_iterations=90,
                batch_size=64,
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
