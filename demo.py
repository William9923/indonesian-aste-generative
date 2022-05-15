from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from colorama import Fore, Style
import torch
import yaml

from src.utility import Path
from src.utility import get_config, set_seed, get_device

from src.postprocess import (
    IPostprocess,
    EditDistancePostProcessor,
    EmbeddingDistancePostProcessor,
)
from src.loader import ILoader, HotelLoader
from src.constant import Path, ModelType, PostprocessType, ProcessType
from src.trainer import ITrainer, T5Trainer
from src.generator import IGenerator, T5Generator

MODEL_NAME = "Wikidepia/IndoT5-base"
CHECKPOINT = "/content/drive/MyDrive/TA/posttraining/checkpoint-10990"

# == Dependencies Maps (Factory) ==
trainer_config_maps = {ModelType.T5Model: T5Trainer}

tokenizer_config_names = {ModelType.T5Model: T5Tokenizer}

generator_config_names = {ModelType.T5Model: T5Generator}

postprocess_config_names = {
    PostprocessType.EDITDISTANCE: EditDistancePostProcessor,
    PostprocessType.EMBEDDING: EmbeddingDistancePostProcessor,
}


def visualize_triplet_opinion(predictions):
    details_accumulators = []
    triplets = []
    NEWLINE = "\n"
    for pred in predictions:
        if pred[2] == "netral":
            color = f"{Fore.YELLOW}"
            symbol = "\u2796"
        elif pred[2] == "negatif":
            color = f"{Fore.RED}"
            symbol = "\u2716"
        else:
            color = f"{Fore.GREEN}"
            symbol = "\u2714"
       
        triplets.append([pred[0], pred[1], pred[2], symbol, color])

    details_accumulators.append("------------------------------------------------------------------------")
    details_accumulators.append("Aspect                         Sentiment                      Polarity")
    details_accumulators.append("========================================================================")

    for _, val in enumerate(triplets):
        details_accumulators.append(
            "%-30s %-30s %-10s"
            % (
                val[0],
                val[1],
                "{} {}".format(val[4], val[2], val[3]),
            )
        )
    details_accumulators.append("------------------------------------------------------------------------")
    for val in details_accumulators:
        print(val)
        print(Style.RESET_ALL)


def predict(generator, sent, fix=True):
    res = generator.generate([sent], fix=fix)
    data = res[0]
    return data


def build_generator(configs, path):
    set_seed(configs["main"]["seed"])
    device = get_device()
    model_type = configs.get("type")
    model_name = configs.get("main").get("pretrained")
    tokenizer = tokenizer_config_names.get(model_type).from_pretrained(model_name)

    checkpoint = torch.load(path, map_location=device)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    postprocessor_type = configs.get("normalization").get("mode")
    postprocessor: IPostprocess = postprocess_config_names.get(postprocessor_type)()

    if isinstance(postprocessor, EmbeddingDistancePostProcessor) and isinstance(
        model, T5ForConditionalGeneration
    ):
        postprocessor.set_embedding(tokenizer, model.get_input_embeddings())

    generator: IGenerator = generator_config_names.get(model_type)(
        tokenizer, model, postprocessor, configs
    )
    return generator


def get_config(path):
    return yaml.safe_load(open(path, "r"))
