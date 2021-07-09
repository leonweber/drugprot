import os
from typing import Optional, List

import bioc
import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, ConcatDataset

from drugprot import utils

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
    if "seed" in config:
        pl.seed_everything(config.seed, workers=True)

    log.info(config)

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")

    model: LightningModule = hydra.utils.instantiate(config.model)

    if config["data"]["checkpoint"]:
        model = model.load_from_checkpoint(
            hydra.utils.to_absolute_path(config["data"]["checkpoint"]),
            **config["model"]
        )

    trainer = None
    if config["data"]["train"]:
        train_data = ConcatDataset(
            [
                model.get_dataset(i, limit_examples=config["data"]["limit_examples"])
                for i in config["data"]["train"]
            ]
        )
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

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=model,
            trainer=trainer,
            callbacks=callbacks,
        )

        train_loader = DataLoader(
            train_data,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=model.collate_fn,
        )
        dev_loader = DataLoader(
            dev_data,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=model.collate_fn,
        )

        model.num_training_steps = len(train_loader) * trainer.max_epochs

        # Train the model
        log.info("Starting training!")
        trainer.fit(
            model=model, train_dataloader=train_loader, val_dataloaders=dev_loader
        )
        best_score = trainer.callback_metrics[config["optimized_metric"]]
        log.info(
            f"Best {config['optimized_metric']} (before finetuning): {best_score})"
        )

        if (
            "finetune" in config["data"]
            and config["data"]["finetune"]
            and config["finetune_trainer"]
        ):
            finetune_data = ConcatDataset(
                [model.get_dataset(i) for i in config["data"]["finetune"]]
            )
            train_loader = DataLoader(
                finetune_data,
                batch_size=config["batch_size"],
                shuffle=True,
                collate_fn=model.collate_fn,
            )
            finetune_trainer: pl.Trainer = hydra.utils.instantiate(
                config.trainer,
                callbacks=callbacks,
                logger=logger,
                _convert_="partial",
            )

            finetune_trainer.fit(model=model, train_dataloader=train_loader)
            best_score = finetune_trainer.callback_metrics[config["optimized_metric"]]
            log.info(
                f"Best {config['optimized_metric']} (after finetuning): {best_score})"
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
        if trainer:
            model = model.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path, **config["model"]
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


if __name__ == "__main__":
    train()
