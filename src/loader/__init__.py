from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from src.loader.parser import parse
from src.loader.dataset import HotelDataset

from src.constant import Path


class Loader:
    def __init__(self, configs):
        self.configs = configs
        self.tokenizer = None
        self.is_loaded = False

        if self.configs["type"] == "t5":
            model_name = self.configs.get("main").get("pretrained")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        self.__load()
        self.is_loaded = True

    def get_train_dataset(self):
        if not self.is_loaded:
            raise ValueError("Loader had not loaded the dataset!")
        return self.train_dataset
    
    def get_test_dataset(self):
        if not self.is_loaded:
            raise ValueError("Loader had not loaded the dataset!")
        return self.test_dataset

    def get_val_dataset(self):
        if not self.is_loaded:
            raise ValueError("Loader had not loaded the dataset!")
        return self.val_dataset

    def get_train_loader(self):
        dataset = self.get_train_dataset()
        batch_size = self.configs.get("trainer").get("batch_size")
        num_workers = self.configs.get("main").get("num_worker")
        return DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )
    
    def get_test_loader(self):
        dataset = self.get_test_dataset()
        batch_size = self.configs.get("trainer").get("eval_batch_size")
        num_workers = self.configs.get("main").get("num_worker")
        return DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

    def get_val_loader(self):
        dataset = self.get_val_dataset()
        batch_size = self.configs.get("trainer").get("eval_batch_size")
        num_workers = self.configs.get("main").get("num_worker")
        return DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

    def __load(self):
        if self.configs is None:
            raise ValueError("config not initialized")

        mode = self.configs.get("loader").get("mode")

        if mode == "filter":
            train_path = Path.TRAIN_FILTERED_PATH
            test_path = Path.TEST_FILTERED_PATH
            val_path = Path.VAL_FILTERED_PATH
        else:
            train_path = Path.TRAIN_UNFILTERED_PATH
            test_path = Path.TEST_UNFILTERED_PATH
            val_path = Path.VAL_UNFILTERED_PATH

        seperator = self.configs.get("loader").get("seperator")

        train_sents, train_labels = parse(train_path, seperator)
        test_sents, test_labels = parse(test_path, seperator)
        val_sents, val_labels = parse(val_path, seperator)

        train_params = {
            "sents": train_sents,
            "labels": train_labels,
            "tokenizer": self.tokenizer,
            "configs": self.configs,
        }
        self.train_dataset = HotelDataset(**train_params)

        val_params = {
            "sents": val_sents,
            "labels": val_labels,
            "tokenizer": self.tokenizer,
            "configs": self.configs,
        }
        self.val_dataset = HotelDataset(**val_params)

        test_params = {
            "sents": test_sents,
            "labels": test_labels,
            "tokenizer": self.tokenizer,
            "configs": self.configs,
        }
        self.test_dataset = HotelDataset(**test_params)
