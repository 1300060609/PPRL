# -*- coding: utf-8 -*-

"""Constants defined for PPRL."""

from collections import OrderedDict
from typing import Callable, Dict
import numpy as np
from pkg_resources import iter_entry_points

VERSION = '0.0.19'

#: Functions for specifying exotic resources with a given prefix
IMPORTERS: Dict[str, Callable[[str], np.ndarray]] = {
    entry_point.name: entry_point.load()
    for entry_point in iter_entry_points(group='PPRL.data.importer')
}


def get_version() -> str:
    """Get the version."""
    return VERSION


pprl = 'pprl'

# KG embedding model
KG_EMBEDDING_MODEL_NAME = 'kg_embedding_model_name'
EXECUTION_MODE = 'execution_mode'

# Model names
PPRL_NAME='PPRL'


# Output paths
ENTITY_TO_EMBEDDINGS = 'entity_to_embeddings'
EVAL_RESULTS = 'eval_results'
ENTITY_TO_ID = 'entity_to_id'
RELATION_TO_ID = 'relation_to_id'

# Device related
PREFERRED_DEVICE = 'preferred_device'
CPU = 'cpu'
GPU = 'gpu'

# ML params
BATCH_SIZE = 'batch_size'
VOCAB_SIZE = 'vocab_size'
EMBEDDING_DIM = 'embedding_dim'
RELATION_EMBEDDING_DIM = 'relation_embedding_dim'
MARGIN_LOSS = 'margin_loss'
NUM_ENTITIES = 'num_entities'
NUM_RELATIONS = 'num_relations'
NUM_EPOCHS = 'num_epochs'
NUM_OF_HPO_ITERS = 'maximum_number_of_hpo_iters'
LEARNING_RATE = 'learning_rate'
TRAINING_MODE = 'Training_mode'
HPO_MODE = 'HPO_mode'
HYPER_PARAMTER_OPTIMIZATION_PARAMS = 'hyper_optimization_params'
TRAINING_SET_PATH = 'training_set_path'
TEST_SET_PATH = 'test_set_path'
TEST_SET_RATIO = 'test_set_ratio'
NORM_FOR_NORMALIZATION_OF_ENTITIES = 'normalization_of_entities'
SCORING_FUNCTION_NORM = 'scoring_function'

# OPTIMIZER
OPTMIZER_NAME = 'optimizer'
SGD_OPTIMIZER_NAME = 'SGD'
ADAGRAD_OPTIMIZER_NAME = 'Adagrad'
ADAM_OPTIMIZER_NAME = 'Adam'

# Further Constants
SEED = 'random_seed'
OUTPUT_DIREC = 'output_direc'

# Pipeline outcome parameters
TRAINED_MODEL = 'trained_model'
LOSSES = 'losses'
ENTITY_TO_EMBEDDING = 'entity_to_embedding'
RELATION_TO_EMBEDDING = 'relation_to_embedding'
CONFIG = 'configuration'
FINAL_CONFIGURATION = 'final_configuration'
EVAL_SUMMARY = 'eval_summary'
