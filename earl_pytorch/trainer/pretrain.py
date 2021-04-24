from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from torch.optim import Adam
from torch.utils.data import DataLoader

from earl_pytorch import EARL
from earl_pytorch.dataset.dataset import get_dataset

from earl_pytorch.dataset.create_dataset import ReplayCollectionDataset
from earl_pytorch.model import EARLReplayModel, EARLActorCritic


def plot_grad_flow(named_parameters):
    import matplotlib.pyplot as plt
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()


class EARLTrainer:
    def __init__(self, earl, train_dataloader: Optional[DataLoader] = None,
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
        self.model = earl.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for EARL" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.val_data = validation_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr)
        # self.optim_schedule = ScheduledOptim(self.optim, self.earl.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.log_freq = log_freq

        print(self.model)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

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
            self.model.train()
        else:
            str_code = "test"
            self.model.eval()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        losses = {}

        def update_losses(name, pred, label, loss, n_classes):
            name_loss = losses.setdefault(name, {"sum_loss": 0, "sum_correct": 0,
                                                 "n_elements": 0, "confusion_matrix": None})
            name_loss["sum_loss"] += loss.item()
            if pred is None:
                return
            name_loss["sum_correct"] += pred.argmax(dim=1).eq(label).sum().item()
            name_loss["n_elements"] += (label >= 0).sum().item()
            label, pred = label.cpu().flatten(), pred.argmax(dim=1).cpu().flatten()
            conf = confusion_matrix(label, pred, labels=np.arange(0, n_classes))
            current_conf = name_loss["confusion_matrix"]
            if current_conf is None:
                name_loss["confusion_matrix"] = conf
            else:
                current_conf += conf

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            # x, y = data[:4], data[4:]
            x, y = data
            x = [v.float().to(self.device) for v in x]
            y = [v.long().to(self.device)
                 for i, v in enumerate(y)]
            # data = {key: ([v.to(self.device) for v in value] if isinstance(value, list) else value.to(self.device))
            #         for key, value in data.items()}
            for _ in range(1):
                model = self.model
                # 1. forward the model
                if isinstance(model, EARLReplayModel):
                    score, next_touch, boost_collect, demo = self.model(*x)

                    # 2-1. Loss of goal classification result
                    score_loss = self.criterion(score, y[0])
                    update_losses("score", score, y[0], score_loss, 2)

                    # 2-2. Loss of predicting next touch
                    touch_loss = self.criterion(next_touch, y[1])
                    n_players = x[2].shape[1] + x[3].shape[1]
                    update_losses("touch", score, y[1], touch_loss, n_players)

                    # 2-3. Loss of predicting next boost collect
                    boost_loss = self.criterion(boost_collect, y[2])
                    n_boosts = x[1].shape[1]
                    update_losses("boost", score, y[2], boost_loss, n_boosts)

                    # 2-4. Loss of predicting next demo
                    demo_loss = self.criterion(demo, y[3])
                    update_losses("demo", score, y[3], demo_loss, n_players)

                    loss = 0.3 * score_loss + 0.5 * touch_loss + 0.1 * boost_loss + 0.1 * demo_loss
                elif isinstance(model, EARLActorCritic):
                    value_pred, actions_pred = self.model(*x)
                    value_target = y[0].float()
                    value_target[value_target < 0] = 0.5
                    value_loss = self.bce(value_pred, value_target)
                    update_losses("value", torch.eye(2)[(value_pred > 0).long()], y[0].cpu(), value_loss, 2)

                    loss = value_loss
                    for j, (name, pred) in enumerate(actions_pred.items(), start=4):
                        target = y[j]
                        action_loss = self.criterion(pred, target)
                        loss += action_loss
                        update_losses(name, pred, y[j], action_loss, 3 if j <= 8 else 2)

                    loss = (value_loss + loss) / (1 + len(actions_pred))
                else:
                    raise ValueError
                update_losses("total", None, None, loss, None)

                # 3. backward and optimization only in train
                if train:
                    self.optim.zero_grad()
                    loss.backward()

                    self.optim.step()

                post_fix = {
                    "epoch": epoch,
                    "iter": i
                }

                post_fix.update({
                    key: {"avg_loss": losses[key]["sum_loss"] / (i + _ + 1),
                          "avg_acc": losses[key]["sum_correct"] / (losses[key]["n_elements"] or 1) * 100}
                    for key in losses})

                data_iter.set_postfix(post_fix)
                if i % self.log_freq == -1 or i == len(data_loader) - 1:
                    import matplotlib.pyplot as plt
                    for k in losses:
                        if losses[k]["confusion_matrix"] is None:
                            continue
                        conf = losses[k]["confusion_matrix"]
                        ConfusionMatrixDisplay(conf).plot(include_values=False)
                        plt.savefig(f"../../out/conf/{k}-{str_code}-{e}.png")
                        plt.close()
                        if not train:
                            print(k)
                            for row in conf:
                                print(list(row / conf.sum()))
        return losses["total"]["sum_loss"] / len(data_iter)

    def save(self, epoch, file_path="../../out/models/{}_trained.model"):
        """
        Saving the current EARL model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path.format(type(self.model).__name__) + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


def dataset_iterator(folder, lim):
    import gc
    for n in range(lim):
        ds = ReplayCollectionDataset(folder, lim)
        is_train = n > 1
        print(n)
        yield DataLoader(ds, batch_size=128, num_workers=0, shuffle=True, drop_last=is_train)
        del ds
        gc.collect()


if __name__ == '__main__':
    # rp_ds = ReplayDatasetFull(None, r"E:\processed", name="data", limit=8192)  # r"E:\replays\ranked-standard"
    # rp_ds = ReplayDatasetFull(None, r"E:\processed", name="data", limit=8192)  # r"E:\replays\ranked-standard"
    # mdl = EARLActorCritic(EARL())
    mdl = torch.load("../../out/models/earl_trained.model.ep2")
    # scr = torch.jit.script(mdl)
    et = EARLTrainer(mdl, lr=3e-4,
                     log_freq=1024, with_cuda=True)
    best_loss = 1e10
    pat = 0
    for e in range(100):
        datasets = dataset_iterator(r"D:\rokutleg\datasets\rlcsx", 11)
        next(datasets)
        et.val_data = next(datasets)
        for ds in datasets:
            et.train_data = None
            et.train_data = ds
            train_loss = et.train(e)

        val_loss = et.validate(e)
        if val_loss < best_loss:
            best_loss = val_loss
            print("New best loss:", best_loss)
            et.save(e)
        else:
            pat += 1
            if pat >= 10:
                break
    et.test_data = next(dataset_iterator(r"D:\rokutleg\datasets\rlcsx", 11))
    et.test(-1)
