from src.trainer.t5 import T5Trainer

# Trainer Interface for fine-tuning pretrain model, can be used for other pre-trained model types... 
# Current Implementation: T5Trainer
# Trainer Interface:
# - fit
# - save
# - load
# - training_report
# - get_model

# Model Interface contains possible method for fine-tuned model...
# Model (fine-tuned) Inteface:
# - generate