from argparse import ArgumentParser
from robust_detection import wandb_config
from robust_detection.data_utils.baselines_data_utils import ObjectsCountDataModule
from robust_detection.baselines.cnn_model import CNN

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(model_cls, data_cls, args):

    if args.paired_og:
        if args.data_dir == "mnist/alldigits":
            args.og_data_dir = "mnist/skip789"
        elif args.data_dir == "mnist/alldigits_2":
            args.og_data_dir = "mnist/skip789_2"
        elif args.data_dir == "mnist/alldigits_5":
            args.og_data_dir = "mnist/skip789_5"
        elif args.data_dir == "mnist/alldigits_20":
            args.og_data_dir = "mnist/skip789_20"
        elif args.data_dir == "molecules/molecules_all":
            args.og_data_dir = "molecules/molecules_skip"
        elif args.data_dir == "clevr/clevr_all":
            args.og_data_dir == "clevr/clevr_skip_cube"
        elif args.data_dir == "mnist/mnist3_all":
            args.og_data_dir = "mnist/mnist3_skip"
        else:
            raise("Invalid data dir name for paired og")


    dataset = data_cls(**vars(args))
    dataset.prepare_data()
    model = model_cls(**vars(args))

    logger = WandbLogger(
        name=f"CNN",
        project="object_detection",
        log_model=False
    )
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.experiment.dir,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=50)

    trainer = pl.Trainer(gpus = args.gpus, logger = logger, callbacks = [checkpoint_cb, early_stopping_cb], max_epochs = args.max_epochs)
    trainer.fit(model, datamodule = dataset)

    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False, gpus = args.gpus)

    model = model_cls.load_from_checkpoint(
        checkpoint_path)
    val_results = trainer2.test(
        model,
        dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model,
        dataloaders=dataset.test_dataloader()
    )[0]

    for name, value in {**test_results}.items():
        logger.experiment.summary['restored_' + name] = value
    for name, value in {**val_results}.items():
        logger.experiment.summary['restored_' + name] = value

if __name__=="__main__":
    
    parser = ArgumentParser()

    # figure out which model to use and other basic params
    parser.add_argument('--fold', default=0, type=int, help=' fold number to use')
    parser.add_argument('--gpus', default=1, type=int, help='the number of gpus to use to train the model')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)

    partial_args, _ = parser.parse_known_args()

    model_cls = CNN
    data_cls = ObjectsCountDataModule 

    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    #if args.T_mask == -1:
    #    args.T_mask = args.T_cond

    main(model_cls, data_cls, args)
