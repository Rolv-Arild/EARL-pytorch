from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from torch.optim import Adam
from torch.utils.data import DataLoader

from earl.dataset.dataset import get_dataset


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
    def __init__(self, earl, train_dataloader: DataLoader, validation_dataloader: Optional[DataLoader] = None,
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

        losses = {k: {"sum_loss": 0, "sum_correct": 0, "n_elements": 0, "confusion_matrix": None}
                  for k in ("total", "score", "touch", "boost", "demo", "ground", "aerial")}

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            x, y = data[:4], data[4:]
            x = [v.float().to(self.device) for v in x]
            y = [v.float().to(self.device)
                 if i == 0 else
                 v.long().to(self.device)
                 for i, v in enumerate(y)]
            # data = {key: ([v.to(self.device) for v in value] if isinstance(value, list) else value.to(self.device))
            #         for key, value in data.items()}
            for _ in range(1):
                # 1. forward the model
                # score, next_touch, boost_collect, demo = self.model(*x)
                score, ground, aerial = self.model(*x)

                # 2-1. Loss of goal classification result
                # score_loss = self.criterion(score, y[0])
                score_loss = self.bce(score, y[0])

                # 2-2. Loss of predicting next touch
                # touch_loss = self.criterion(next_touch, y[2])

                # 2-3. Loss of predicting next boost collect
                # boost_loss = self.criterion(boost_collect, y[3])

                # 2-4. Loss of predicting next demo
                # demo_loss = self.criterion(demo, y[4])

                # 2-5. Loss of predicting next action
                ground_loss = self.criterion(ground, y[5])
                aerial_loss = self.criterion(aerial, y[6])

                # 2-5. Adding losses
                # loss = 0.3 * score_loss + 0.5 * touch_loss + 0.1 * boost_loss + 0.1 * demo_loss
                loss = (score_loss + ground_loss + aerial_loss) / 3

                # 3. backward and optimization only in train
                if train:
                    # self.optim_schedule.zero_grad()
                    self.optim.zero_grad()
                    loss.backward()

                    # plot_grad_flow(self.model.named_parameters())
                    # exit(0)

                    self.optim.step()
                    # self.optim_schedule.step_and_update_lr()

                losses["total"]["sum_loss"] += loss.item()

                losses["score"]["sum_loss"] += score_loss.item()
                losses["score"]["sum_correct"] += (score > 0).eq(y[0]).sum().item()
                losses["score"]["n_elements"] += (y[0] >= 0).sum().item()
                score_conf = losses["score"]["confusion_matrix"]
                label, pred = y[0].cpu(), (score > 0).cpu()
                conf = confusion_matrix(label, pred, labels=np.arange(0, 2))
                if score_conf is None:
                    losses["score"]["confusion_matrix"] = conf
                else:
                    score_conf += conf

                # losses["touch"]["sum_loss"] += touch_loss.item()
                # losses["touch"]["sum_correct"] += next_touch.argmax(dim=1).eq(y[2]).sum().item()
                # losses["touch"]["n_elements"] += (y[2] >= 0).sum().item()
                #
                # losses["boost"]["sum_loss"] += boost_loss.item()
                # losses["boost"]["sum_correct"] += boost_collect.argmax(dim=1).eq(y[3]).sum().item()
                # losses["boost"]["n_elements"] += (y[3] >= 0).sum().item()
                #
                # losses["demo"]["sum_loss"] += demo_loss.item()
                # losses["demo"]["sum_correct"] += demo.argmax(dim=1).eq(y[4]).sum().item()
                # losses["demo"]["n_elements"] += (y[4] >= 0).sum().item()

                losses["ground"]["sum_loss"] += ground_loss.item()
                losses["ground"]["sum_correct"] += ground.argmax(dim=1).eq(y[5]).sum().item()
                losses["ground"]["n_elements"] += (y[5] >= 0).sum().item()
                ground_conf = losses["ground"]["confusion_matrix"]
                label, pred = y[5].flatten().cpu(), ground.argmax(dim=1).flatten().cpu()
                conf = confusion_matrix(label, pred, labels=np.arange(0, 72))
                if ground_conf is None:
                    losses["ground"]["confusion_matrix"] = conf
                else:
                    ground_conf += conf

                losses["aerial"]["sum_loss"] += aerial_loss.item()
                losses["aerial"]["sum_correct"] += aerial.argmax(dim=1).eq(y[6]).sum().item()
                losses["aerial"]["n_elements"] += (y[6] >= 0).sum().item()
                aerial_conf = losses["aerial"]["confusion_matrix"]
                label, pred = y[6].flatten().cpu(), aerial.argmax(dim=1).flatten().cpu()
                conf = confusion_matrix(label, pred, labels=np.arange(0, 108))
                if aerial_conf is None:
                    losses["aerial"]["confusion_matrix"] = conf
                else:
                    aerial_conf += conf

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
                        print(k)
                        conf = losses[k]["confusion_matrix"]
                        ConfusionMatrixDisplay(conf).plot(include_values=False)
                        plt.show()
                        for row in conf:
                            print(list(row / conf.sum()))
        return losses["total"]["sum_loss"] / len(data_iter)

    def save(self, epoch, file_path="output/earl_trained.model"):
        """
        Saving the current EARL model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


if __name__ == '__main__':
    # rp_ds = ReplayDatasetFull(None, r"E:\processed", name="data", limit=8192)  # r"E:\replays\ranked-standard"
    # rp_ds = ReplayDatasetFull(None, r"E:\processed", name="data", limit=8192)  # r"E:\replays\ranked-standard"
    rp_ds = get_dataset(None, r"E:\processed", name="dataset", limit=4096)
    lens = [70 * len(rp_ds) // 100,
            15 * len(rp_ds) // 100,
            15 * len(rp_ds) // 100]
    diff = len(rp_ds) - sum(lens)
    lens[0] += diff
    train, test, val = torch.utils.data.random_split(rp_ds, lens)
    # exit(0)
    # ds = BufferedShuffleDataset(rp_ds, 65536)
    # dl = DataLoader(rp_ds, batch_size=512, num_workers=0)
    train_dl = DataLoader(train, batch_size=128, num_workers=0, shuffle=True)
    test_dl = DataLoader(test, batch_size=128, num_workers=0)
    val_dl = DataLoader(val, batch_size=128, num_workers=0)

    # mdl = EARLActorCritic(EARL())
    mdl = torch.load("../../out/models/EARL")
    # scr = torch.jit.script(mdl)
    et = EARLTrainer(mdl, lr=3e-4, train_dataloader=train_dl, validation_dataloader=val_dl, test_dataloader=test_dl,
                     log_freq=1024, with_cuda=True)
    # best_loss = 1e10
    # pat = 0
    # for e in range(100):
    #     train_loss = et.train(e)
    #     val_loss = et.validate(e)
    #     if val_loss < best_loss:
    #         best_loss = val_loss
    #         print("New best loss:", best_loss)
    #         et.save(e)
    #     else:
    #         pat += 1
    #         if pat >= 10:
    #             break
    et.test(0)
