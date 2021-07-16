import hydra
import comet_ml
import pytorch_lightning as pl
import pandas as pd
import pickle

from typing import Optional, List
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pathlib import Path

from torch.utils.data import DataLoader

from drugprot import utils
from drugprot.models.entity_classifier_baseline import EntityInteractionDataSet
from drugprot.utils import read_vocab

log = utils.get_logger(__name__)


@hydra.main(config_path="../configs/entity_classifier", config_name="config.yaml")
def train_classifier(config: DictConfig) -> Optional[float]:
    if "seed" in config:
        pl.seed_everything(config.seed, workers=True)

    log.info(config)


    entity_to_id = read_vocab(Path(config["embeddings"]["dict_file"]))
    entity_embeddings = pickle.load(Path(config["embeddings"]["emb_file"]).open("rb"))
    entity_embeddings = entity_embeddings.float()

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
        embeddings=entity_embeddings
    )

    print(model)

    if config["data"]["checkpoint"]:
        model = model.load_from_checkpoint(
            hydra.utils.to_absolute_path(config["data"]["checkpoint"]),
            **config["model"]
        )

    if config["data"]["train_file"]:
        # Load train and dev data
        train_data_raw = pd.read_csv(config["data"]["train_file"], sep="\t")
        train_data = EntityInteractionDataSet(train_data_raw, entity_to_id)
        log.info(f"Training data set contains {len(train_data)} instances")

        dev_data_raw = pd.read_csv(config["data"]["dev_file"], sep="\t")
        dev_data = EntityInteractionDataSet(dev_data_raw, entity_to_id)
        log.info(f"Dev data set contains {len(dev_data)} instances")

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
        )

        train_loader = DataLoader(
            train_data,
            batch_size=config["batch_size"],
            shuffle=True
        )

        dev_loader = DataLoader(
            dev_data,
            batch_size=config["batch_size"],
            shuffle=False
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


if __name__ == "__main__":
    train_classifier()
