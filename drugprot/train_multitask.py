import os
from pathlib import Path
from typing import Optional, List

import bioc
import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, ConcatDataset

from drugprot import utils
from drugprot.multitask_data import MultiTaskDataset, MultiTaskBatchSampler

log = utils.get_logger(__name__)


@hydra.main(config_path="../configs/", config_name="config.yaml")
def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    best_score = 0.0

    if "seed" in config:
        pl.seed_everything(config.seed, workers=True)

    log.info(config)

    dataset_to_meta = {}
    for data_path in config["data"]["train"]:
        meta = utils.get_dataset_metadata(data_path)
        dataset_to_meta[meta.name] = meta

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model,
                                                     dataset_to_meta=dataset_to_meta)
    if config["checkpoint"]:
        checkpoint = config["checkpoint"]
        model = model.load_from_checkpoint(
            hydra.utils.to_absolute_path(checkpoint),
            dataset_to_meta=dataset_to_meta,
            strict=False,
            **config["model"]
        )

    if config["trainer"]["_target_"] != "None":
        train_datasets = [
            model.get_dataset(i, limit_examples=config["data"]["limit_examples"])
            for i in config["data"]["train"]
        ]
        dev_data = model.get_dataset(
            config["data"]["dev"], limit_examples=config["data"]["limit_examples"]
        )

        # Init Lightning callbacks
        callbacks: List[pl.Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        # Init Lightning loggers
        logger: List[pl.LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config["logger"].items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

        # Init Lightning trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer: pl.Trainer = hydra.utils.instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )

        log.info("Initializing")
        utils.init(
            config=config,
            model=model,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        utils.log_hyperparameters(
            config=config,
            model=model,
            trainer=trainer,
        )

        train_dataset = MultiTaskDataset(datasets=train_datasets)

        train_batch_sampler = MultiTaskBatchSampler(datasets=train_datasets,
                                                    dataset_to_batch_size=config["data"]["dataset_to_batch_size"],
                                                    mix_opt=0,
                                                    extra_task_ratio=0)

        train_loader = DataLoader(
            dataset=train_dataset,
            collate_fn=model.collate_fn,
            batch_sampler=train_batch_sampler
        )

        dev_dataset = MultiTaskDataset(datasets=[dev_data])
        dev_batch_sampler = MultiTaskBatchSampler(datasets=[dev_data],
                                                  dataset_to_batch_size=config["data"]["dataset_to_batch_size"],
                                                 mix_opt=0,
                                                 extra_task_ratio=0)

        dev_loader = DataLoader(
            dataset=dev_dataset,
            collate_fn=model.collate_fn,
            batch_sampler=dev_batch_sampler
        )

        model.num_training_steps = len(train_loader) * trainer.max_epochs

        # Train the model
        log.info("Starting training!")
        trainer.fit(
            model=model, train_dataloader=train_loader, val_dataloaders=dev_loader
        )
        best_score = trainer.callback_metrics[config["model"]["optimized_metric"]]
        log.info(
            f"Best {config['model']['optimized_metric']} (before finetuning): {best_score})"
        )
        # Make sure everything closed properly
        utils.finish(
            config=config,
            model=model,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

    if (
        config["finetune_trainer"]["_target_"] != "None"
    ):
        model.lr = model.finetune_lr
        finetune_datasets = [
            model.get_dataset(i, limit_examples=config["data"]["limit_examples"])
            for i in config["data"]["finetune"]
        ]
        finetune_dataset = MultiTaskDataset(datasets=train_datasets)
        finetune_batch_sampler = MultiTaskBatchSampler(datasets=finetune_datasets,
                                                       dataset_to_batch_size=config["data"]["dataset_to_batch_size"],
                                                       mix_opt=0,
                                                       extra_task_ratio=0)
        # Init Lightning loggers
        logger: List[pl.LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config["logger"].items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

        train_loader = DataLoader(
            dataset=finetune_dataset,
            collate_fn=model.collate_fn,
            batch_sampler=finetune_batch_sampler
        )
        callbacks: List[pl.Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))
        finetune_trainer: pl.Trainer = hydra.utils.instantiate(
            config.finetune_trainer,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )

        finetune_trainer.fit(model=model, train_dataloader=train_loader,
                             val_dataloaders=dev_loader)
        best_score = finetune_trainer.callback_metrics[config["model"]["optimized_metric"]]
        log.info(
            f"Best {config['model']['optimized_metric']} (after finetuning): {best_score})"
        )

        # Make sure everything closed properly
        log.info("Finalizing!")
        utils.finish(
            config=config,
            model=model,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Print path to best checkpoint
        log.info(
            f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}"
        )

    if "predict" in config["data"] and config["data"]["predict"]:
        try:
            model = model.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path, **config["model"],
                dataset_to_meta=dataset_to_meta
            )
        except NameError:
            model = model.load_from_checkpoint(
                hydra.utils.to_absolute_path(config.data.checkpoint), **config["model"],
                dataset_to_meta=dataset_to_meta
            )
        if torch.cuda.is_available():
            model.cuda()
        with open(hydra.utils.to_absolute_path(config["data"]["predict"])) as f:
            with torch.no_grad():
                model.eval()
                collection = bioc.load(f)
                model.predict(collection)

            with open("predictions.bioc.xml", "w") as f:
                bioc.dump(collection, f)
            log.info(f"Wrote predictions to {os.path.abspath(os.getcwd())}/predictions.bioc.xml")

    return best_score


if __name__ == "__main__":
    train()
