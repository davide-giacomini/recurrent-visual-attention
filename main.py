import sys
from bayes_opt import BayesianOptimization
import torch
from LOI import LookupOnlyInference

import utils
import data_loader

from trainer import Trainer
from config import get_config


def main(config):
    utils.prepare_dirs(config)


    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    ram_data_loader = data_loader.RAMDataLoader(config.dataset, config.data_dir)

    # instantiate data loaders
    if config.is_train:
        dloader = ram_data_loader.get_train_valid_loader(
            batch_size=config.batch_size,
            random_seed=config.random_seed,
            valid_size=config.valid_size,
            shuffle=config.shuffle,
            show_sample=config.show_sample,
            **kwargs,
        )
    elif config.is_train_table:
        dloader = ram_data_loader.get_train_table_loader(
            batch_size=config.batch_size,
            **kwargs,
        )
    else:
        dloader = ram_data_loader.get_test_loader(
            batch_size=config.batch_size,
            **kwargs,
        )

    if config.mem_based_inference:
        if config.is_train:
            print("Error: config.is_train==True -- If the look-up only flag is active, the model cannot be in train mode.")
            sys.exit(1)
        
        if config.is_train_table:
            pass #TODO complete

        dloader = ram_data_loader.get_test_loader(
            batch_size=config.batch_size,
            **kwargs,
        )

        lookup_inference = LookupOnlyInference(config, dloader)

    trainer = Trainer(config, dloader)

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
    # or load a pretrained model and test
    elif config.is_train_table:
        trainer.prepare_training_table()
    elif config.mem_based_inference:    
        if config.bo:
            trainer.BO()
        else:
            trainer.memory_based_inference(a=config.a, b=config.b, c=config.c)
    else:
        import time
        start_test = time.time()
        trainer.test()
        end_test = time.time()
        print("Test time: ", end_test-start_test)


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
