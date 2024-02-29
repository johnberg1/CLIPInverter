from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
  'celeba_encode': {
    'transforms': transforms_config.EncodeTransforms,
    'train_source_root': dataset_paths['celeba_train'],
    'train_target_root': dataset_paths['celeba_train'],
    'test_source_root': dataset_paths['celeba_test'],
    'test_target_root': dataset_paths['celeba_test'],
  },
  'afhq_encode': {
    'transforms': transforms_config.EncodeTransforms,
    'train_source_root': dataset_paths['afhq_train'],
    'train_target_root': dataset_paths['afhq_train'],
    'test_source_root': dataset_paths['afhq_test'],
    'test_target_root': dataset_paths['afhq_test'],
  },
  'cub_encode': {
    'transforms': transforms_config.CUBEncodeTransforms,
    'train_source_root': dataset_paths['cub_train'],
    'train_target_root': dataset_paths['cub_train'],
    'test_source_root': dataset_paths['cub_test'],
    'test_target_root': dataset_paths['cub_test'],
  }
}
