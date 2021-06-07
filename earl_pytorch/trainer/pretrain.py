import time
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning
import sklearn.metrics
import torch
import torch.nn as nn
import tqdm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, BufferedShuffleDataset
import contextlib
from earl_pytorch import EARL
from earl_pytorch.dataset.dataset import get_dataset
import pytorch_lightning as pl
import torchmetrics

from earl_pytorch.dataset.create_dataset import ReplayCollectionDataset
from earl_pytorch.dataset.fast_tensor_data_loader import FastTensorDataLoader
from earl_pytorch.model import EARLReplayModel, EARLActorCritic


class EARLTrainer:
    def __init__(self, models, train_dataloader: Optional[DataLoader] = None,
                 validation_dataloader: Optional[DataLoader] = None,
                 test_dataloader: Optional[DataLoader] = None, lr: float = 3e-4, betas=(0.9, 0.999),
                 weight_decay: float = 0.01, with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        # Setup cuda device for EARL training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This EARL model will be saved every epoch
        # self.earl = earl.to(self.device)
        # Initialize the EARL Replay Model, with EARL model
        # self.model = EARLReplayModel(earl).to(self.device)
        if isinstance(models, list):
            self.models = [m.to(self.device) for m in models]
        else:
            self.models = [models.to(self.device)]

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for EARL" % torch.cuda.device_count())
            self.models = [nn.DataParallel(m, device_ids=cuda_devices) for m in self.models]

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.val_data = validation_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optims = [Adam(m.parameters(), lr=lr) for m in self.models]
        # self.optim = Adam(self.model.parameters(), lr=lr)
        # self.optim_schedule = ScheduledOptim(self.optim, self.earl.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.cel = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.log_freq = log_freq

        for m in self.models:
            print(m)
        print("Total Parameters:", [sum(p.nelement() for p in m.parameters()) for m in self.models])

    def train(self, epoch):
        return self.iteration(epoch, self.train_data)

    def test(self, epoch):
        return self.iteration(epoch, self.test_data, train=False)

    def validate(self, epoch):
        return self.iteration(epoch, self.val_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        if train:
            str_code = "train"
            for model in self.models:
                model.train()
            context = contextlib.nullcontext()
        else:
            str_code = "test"
            for model in self.models:
                model.eval()
            context = torch.no_grad()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        all_losses = [{"total": {"sum_loss": 0, "sum_correct": 0,
                                 "n_elements": 0, "confusion_matrix": None}} for _ in self.models]

        def update_losses(model_index, out_name, predictions, labels, loss_value, n_classes):
            name_loss = all_losses[model_index].setdefault(out_name, {"sum_loss": 0, "sum_correct": 0,
                                                                      "n_elements": 0, "confusion_matrix": None})
            name_loss["sum_loss"] += loss_value.cpu().item()
            if predictions is None:
                return
            name_loss["sum_correct"] += predictions.argmax(dim=1).eq(labels).sum().item()
            name_loss["n_elements"] += (labels >= 0).sum().item()
            labels, predictions = labels.cpu().flatten(), predictions.argmax(dim=1).cpu().flatten()
            conf = confusion_matrix(labels, predictions, labels=np.arange(0, n_classes))
            current_conf = name_loss["confusion_matrix"]
            if current_conf is None:
                name_loss["confusion_matrix"] = conf
            else:
                current_conf += conf

        with context:
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                # x, y = data[:4], data[4:]
                x, y = data
                x = [v.float().to(self.device) for v in x]
                y = [v.long().to(self.device)
                     for i, v in enumerate(y)]
                # data = {key: ([v.to(self.device) for v in value] if isinstance(value, list) else value.to(self.device))
                #         for key, value in data.items()}
                for m, (model, optim) in enumerate(zip(self.models, self.optims)):
                    # 1. forward the model
                    if isinstance(model, EARLReplayModel):
                        score, next_touch, boost_collect, demo = model(*x)

                        # 2-1. Loss of goal classification result
                        score_loss = self.cel(score, y[0])
                        update_losses(m, "score", score, y[0], score_loss, 2)

                        # 2-2. Loss of predicting next touch
                        touch_loss = self.cel(next_touch, y[1])
                        n_players = x[2].shape[1]
                        update_losses(m, "touch", next_touch, y[1], touch_loss, n_players)

                        # 2-3. Loss of predicting next boost collect
                        boost_loss = self.cel(boost_collect, y[2])
                        n_boosts = x[1].shape[1]
                        update_losses(m, "boost", boost_collect, y[2], boost_loss, n_boosts)

                        # 2-4. Loss of predicting next demo
                        demo_loss = self.cel(demo, y[3])
                        update_losses(m, "demo", demo, y[3], demo_loss, n_players)

                        loss = 0.3 * score_loss + 0.5 * touch_loss + 0.1 * boost_loss + 0.1 * demo_loss
                        update_losses(m, "total", None, None, loss, None)
                    elif isinstance(model, EARLActorCritic):
                        value_pred, actions_pred = model(*x)
                        value_target = y[0].float()
                        value_target[value_target < 0] = 0.5
                        value_loss = self.bce(value_pred, value_target)
                        update_losses(m, "value", torch.eye(2)[(value_pred > 0).long()], y[0].cpu(), value_loss, 2)

                        loss = value_loss
                        for j, (name, pred) in enumerate(actions_pred.items(), start=4):
                            target = y[j]
                            action_loss = self.cel(pred, target)
                            loss += action_loss
                            update_losses(m, f"control_{name}", pred, y[j], action_loss, 3 if j <= 8 else 2)

                        loss = (value_loss + loss) / (1 + len(actions_pred))
                        update_losses(m, "total", None, None, loss, None)
                    else:
                        raise ValueError

                    # 3. backward and optimization only in train
                    if train:
                        optim.zero_grad()
                        loss.backward()

                        optim.step()

                if i % self.log_freq == -1 or i == len(data_loader) - 1:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i
                    }

                    post_fix.update({repr(muddel): {
                        key: {"avg_loss": losses[key]["sum_loss"] / (i + 1),
                              "avg_acc": losses[key]["sum_correct"] / (losses[key]["n_elements"] or 1) * 100}
                        for key in losses} for muddel, losses in zip(self.models, all_losses)})

                    # data_iter.set_postfix(post_fix)

                    import json
                    print(json.dumps(post_fix, indent=1))

                    import matplotlib.pyplot as plt
                    for losses in all_losses:
                        for k in losses:
                            if losses[k]["confusion_matrix"] is None:
                                continue
                            conf = losses[k]["confusion_matrix"]
                            ConfusionMatrixDisplay(conf).plot(include_values=False)
                            plt.savefig(f"../../out/conf/{k}-{str_code}-{e}.png")
                            plt.close()
                            # if not train:
                            #     print(k)
                            #     for row in conf:
                            #         print(list(row / conf.sum()))
            total_losses = []
            for losses in all_losses:
                for key in losses:
                    if key == "total":
                        tot = losses[key]["sum_loss"] / len(data_iter)
                        total_losses.append(tot)
            return total_losses

    def save(self, epoch, i, file_path="../../out/models/{}_trained.model"):
        """
        Saving the current EARL model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        model = self.models[i]
        output_path = file_path.format(repr(model)) + ".ep%d" % epoch
        torch.save(model.cpu(), output_path)
        model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


def dataset_iterator(folder, lim, bs=128):
    import gc
    for n in range(lim):
        ds = ReplayCollectionDataset(folder, lim)
        is_train = n > 1
        print(n)
        # yield FastTensorDataLoader(folder, n, bs, shuffle=True)
        yield DataLoader(ds, batch_size=bs, num_workers=0, shuffle=True, drop_last=is_train, pin_memory=True)
        del ds
        gc.collect()


class LitEARLTrainer(pl.LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = 3e-4

        if isinstance(model, EARLReplayModel):
            self.outputs = {
                "score": {
                    "weight": 0.3,
                    "loss_func": nn.CrossEntropyLoss(),
                    "metrics": {
                        "acc": torchmetrics.Accuracy(num_classes=2),
                        "conf": torchmetrics.ConfusionMatrix(num_classes=2)
                    }
                },
                "touch": {
                    "weight": 0.5,
                    "loss_func": nn.CrossEntropyLoss(),
                    "metrics": {
                        "acc": torchmetrics.Accuracy(num_classes=6),
                        "conf": torchmetrics.ConfusionMatrix(num_classes=6)
                    }
                },
                "boost": {
                    "weight": 0.1,
                    "loss_func": nn.CrossEntropyLoss(),
                    "metrics": {
                        "acc": torchmetrics.Accuracy(num_classes=34),
                        "conf": torchmetrics.ConfusionMatrix(num_classes=34)
                    }
                },
                "demo": {
                    "weight": 0.1,
                    "loss_func": nn.CrossEntropyLoss(),
                    "metrics": {
                        "acc": torchmetrics.Accuracy(num_classes=6),
                        "conf": torchmetrics.ConfusionMatrix(num_classes=6)
                    }
                }
            }
        elif isinstance(model, EARLActorCritic):
            self.outputs = {
                "value": {
                    "weight": 1 / 9,
                    "loss_func": nn.BCEWithLogitsLoss(),
                    "metrics": {
                        "acc": torchmetrics.Accuracy(num_classes=2),
                        "conf": torchmetrics.ConfusionMatrix(num_classes=2)
                    }
                }
            }
            self.outputs.update({
                {
                    name: {
                        "weight": 1 / 9,
                        "loss_func": nn.CrossEntropyLoss(),
                        "metrics": {
                            "acc": torchmetrics.Accuracy(num_classes=3),
                            "conf": torchmetrics.ConfusionMatrix(num_classes=3)
                        }
                    }
                }
                for name in ("throttle", "steer", "pitch", "yaw", "roll")
            })
            self.outputs.update({
                {
                    name: {
                        "weight": 1 / 9,
                        "loss_func": nn.CrossEntropyLoss(),
                        "metrics": {
                            "acc": torchmetrics.Accuracy(num_classes=2),
                            "conf": torchmetrics.ConfusionMatrix(num_classes=2)
                        }
                    }
                }
                for name in ("jump", "boost", "handbrake")
            })

    def forward(self, *x):
        return self.model(*x)

    def on_post_move_to_device(self):
        for info in self.outputs.values():
            info["loss_func"].to(self.device)
            for metric in info["metrics"].values():
                metric.to(self.device)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    def _base_step(self, batch, batch_idx, mode):
        x, y = batch
        x = [v.float() for v in x]
        y = [v.long() for v in y]

        y_hat = self.model(*x)

        total_loss = 0
        for name, y_pred, y_true in zip(self.outputs, y_hat, y):
            info = self.outputs[name]
            loss = info["loss_func"](y_pred, y_true)
            self.log(f"{mode}_{name}_loss", loss, on_epoch=True)
            for metric in info["metrics"].values():
                metric(y_pred.argmax(1)[y_true >= 0], y_true[y_true >= 0])
            total_loss += info["weight"] * loss
        self.log(f"{mode}_loss", total_loss)

        return total_loss

    def _base_epoch_end(self, mode):
        for name, info in self.outputs.items():
            for metric_name, metric in info["metrics"].items():
                res = metric.compute()
                if res.numel() > 1:
                    tensorboard = self.logger.experiment
                    df_cm = pd.DataFrame(res.cpu().numpy())
                    cmd = ConfusionMatrixDisplay(df_cm).plot(include_values=False)
                    tensorboard.add_figure(f"{mode}_{name}_{metric_name}_epoch", cmd.figure_, self.current_epoch)
                else:
                    self.log(f"{mode}_{name}_{metric_name}_epoch", res)
                metric.reset()

    def training_step(self, train_batch, batch_idx):
        return self._base_step(train_batch, batch_idx, "train")

    def training_epoch_end(self, outputs):
        self._base_epoch_end("train")

    def validation_epoch_end(self, outputs):
        self._base_epoch_end("val")

    def test_epoch_end(self, outputs):
        self._base_epoch_end("test")

    def validation_step(self, train_batch, batch_idx):
        return self._base_step(train_batch, batch_idx, "val")

    def test_step(self, train_batch, batch_idx):
        return self._base_step(train_batch, batch_idx, "test")


class RLCSXDataset(pytorch_lightning.LightningDataModule):
    def __init__(self, fname, batch_size=128):
        super().__init__()
        self.h5_dataset = h5py.File(fname, "r")
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # if stage == "fit":
        self.train = (self.h5_dataset["train"])
        # elif stage == "validate":
        self.val = (self.h5_dataset["val"])
        # elif stage == "test":
        self.test = (self.h5_dataset["test"])

    def train_dataloader(self):
        rcd = ReplayCollectionDataset(self.train)
        return DataLoader(rcd, batch_size=None, pin_memory=True)

    def val_dataloader(self):
        rcd = ReplayCollectionDataset(self.val)
        # bsd = BufferedShuffleDataset(rcd, 65536)
        return DataLoader(rcd, batch_size=None, pin_memory=True)

    def test_dataloader(self):
        rcd = ReplayCollectionDataset(self.test)
        return DataLoader(rcd, batch_size=None, pin_memory=True)


if __name__ == '__main__':
    # time.sleep(60 * 90)
    pytorch_lightning.seed_everything(123)
    # models = [
    #     EARLReplayModel(EARL(256, 8, 8)),
    #     EARLActorCritic(EARL(256, 8, 8))
    # ]
    model = LitEARLTrainer(EARLReplayModel(EARL(256, 8, 8)))
    logger = TensorBoardLogger(save_dir="../../out/logs", name="EARLReplayModel")
    trainer = pytorch_lightning.Trainer(logger, auto_lr_find=True, gpus=1, max_epochs=100, callbacks=[
        ModelCheckpoint("../../out/models", "EARLReplayModel-{epoch}")])

    # datasets = dataset_iterator(r"D:\rokutleg\datasets\rlcsx", 12, 512)
    dataset = RLCSXDataset(r"D:\rokutleg\datasets\rlcsx.hdf5", 512)

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model, dataset, max_lr=5e-2)
    fig = lr_finder.plot(suggest=True)
    fig.show()
    new_lr = lr_finder.suggestion()
    model.lr = new_lr

    trainer.fit(model, datamodule=dataset)
