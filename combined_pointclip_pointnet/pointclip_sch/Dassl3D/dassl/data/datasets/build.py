from dassl.utils import Registry, check_availability
#from datasets.scanobjnn import ScanObjectNN
DATASET_REGISTRY = Registry('DATASET')
#DATASET_REGISTRY.register(ScanObjectNN)


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)

    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
