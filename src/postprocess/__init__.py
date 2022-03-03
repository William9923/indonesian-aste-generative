from src.postprocess.editdistance import EditDistancePostProcessor
from src.postprocess.embedding import EmbeddingDistancePostProcessor

# Postprocess Interface to check prediction and make normalization to non-correct aspect/sentiment term...
# Current Implementation: EditDistancePostProcessor, EmbeddingDistancePostProcessor
# Postprocess Interface:
# - check_and_fix
# - recover_term