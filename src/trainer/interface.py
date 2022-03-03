# Trainer Interface for fine-tuning pretrain model, can be used for other pre-trained model types...
# Current Implementation: T5Trainer
# Trainer Interface:
# - fit
# - save
# - load
# - training_report
# - get_model
from typing import Tuple, List, Any
from transformers.generation_utils import GenerationMixin


class ITrainer:
    def fit(self):
        pass

    def save(self):
        pass

    def load(self, path: str):
        pass

    def get_model(self) -> GenerationMixin:
        pass

    def training_report(self) -> List[str]:
        pass
