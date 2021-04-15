import gc
import os
import pickle

import numpy as np
import torch
import tqdm
from torch.utils import data as data
from torch.utils.data.dataset import T_co, TensorDataset, ConcatDataset

from earl_pytorch.dataset.create_dataset import replay_to_dfs, convert_dfs, normalize, swap_teams


class ReplayDataset(data.IterableDataset):
    def __init__(self, replay_folder, cache_folder, limit=np.inf):
        self.replay_folder = replay_folder
        self.cache_folder = cache_folder
        self.limit = limit

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.replay_folder is None:
            files = [(dp, f) for dp, dn, fn in os.walk(self.cache_folder) for f in fn if f.endswith(".pickle")]
        else:
            files = [(dp, f) for dp, dn, fn in os.walk(self.replay_folder) for f in fn if f.endswith(".replay")]
        if worker_info is None:
            start = 0
            step = 1
        else:
            start = worker_info.id
            step = worker_info.num_workers

        n = 0
        for dp, f in files[start::step]:
            try:
                if self.replay_folder is None:
                    out_path = os.path.join(dp, f)
                    with open(out_path, "rb") as handle:
                        dfs = pickle.load(handle)
                else:
                    in_path = os.path.join(dp, f)
                    out_path = os.path.join(self.cache_folder, f[:-7] + ".pickle")
                    if os.path.exists(out_path):
                        with open(out_path, "rb") as handle:
                            dfs = pickle.load(handle)
                    else:
                        dfs = replay_to_dfs(in_path)
                        with open(out_path, "wb") as handle:
                            pickle.dump(dfs, handle)

                x_n, y_n = convert_dfs(dfs)
                assert x_n[2].shape == x_n[3].shape
                normalize(x_n)

                yield from zip(zip(*x_n), zip(*y_n))
                swap_teams(x_n, y_n)
                yield from zip(zip(*x_n), zip(*y_n))

                n += len(x_n[0])
                if n >= self.limit:
                    return
            except Exception as e:
                # print(e)
                pass

    def __len__(self):
        return self.limit


class ReplayDatasetFull(data.Dataset):
    def __init__(self, replay_folder, cache_folder, limit=None, name=None):
        self.replay_folder = replay_folder
        self.cache_folder = cache_folder
        self.limit = limit

        if name is not None:
            save_path = os.path.join(cache_folder, f"{name}.npz")
            if os.path.exists(save_path):
                arrs = np.load(save_path)
                arrs = list(arrs.values())
                self.x, self.y = arrs[:4], arrs[4:]
                swap_teams(self.x, self.y, slice(None, None, 2))
                return

        if self.replay_folder is None:
            files = [(dp, f) for dp, dn, fn in os.walk(self.cache_folder) for f in fn if f.endswith(".pickle")]
        else:
            files = [(dp, f) for dp, dn, fn in os.walk(self.replay_folder) for f in fn if f.endswith(".replay")]

        file_iter = tqdm.tqdm(enumerate(files[:limit]),
                              desc="Load",
                              total=limit,
                              bar_format="{l_bar}{r_bar}")

        arrays = []
        for i, (dp, f) in file_iter:
            try:
                if self.replay_folder is None:
                    out_path = os.path.join(dp, f)
                    with open(out_path, "rb") as handle:
                        dfs = pickle.load(handle)
                else:
                    in_path = os.path.join(dp, f)
                    out_path = os.path.join(self.cache_folder, f[:-7] + ".pickle")
                    if os.path.exists(out_path):
                        with open(out_path, "rb") as handle:
                            dfs = pickle.load(handle)
                    else:
                        dfs = replay_to_dfs(in_path)
                        with open(out_path, "wb") as handle:
                            pickle.dump(dfs, handle)

                x_n, y_n = convert_dfs(dfs)
                assert x_n[2].shape == x_n[3].shape
                normalize(x_n)

                arrays.append((x_n, y_n))

                # x_s, y_s = [v.copy() for v in x_n], [v.copy() for v in y_n]
                # swap_teams(x_s, y_s)

                # arrays.append((x_s, y_s))
            except Exception as e:
                print(e)
                pass

        print("Concatenating...")
        self.x = []
        for i in range(len(arrays[0][0])):
            conc = []
            for row in arrays:
                conc.append(row[0][i])
                row[0][i] = None
            self.x.append(np.concatenate(conc))
            conc.clear()
            gc.collect()

        self.y = []
        for i in range(len(arrays[0][1])):
            conc = []
            for row in arrays:
                conc.append(row[1][i])
                row[1][i] = None
            self.y.append(np.concatenate(conc))
            conc.clear()
            gc.collect()

        if name is not None:
            np.savez_compressed(save_path, *self.x, *self.y)
        swap_teams(self.x, self.y, slice(None, None, 2))

    def __getitem__(self, index) -> T_co:
        return [v[index] for v in self.x], [v[index] for v in self.y]

    def __len__(self):
        return self.x[0].shape[0]


def get_dataset(replay_folder, cache_folder, limit, name=None):
    if name is not None:
        name = os.path.join(cache_folder, name)
        if os.path.exists(name):
            return torch.load(name)

    if replay_folder is None:
        files = [(dp, f) for dp, dn, fn in os.walk(cache_folder) for f in fn if f.endswith(".pickle")]
    else:
        files = [(dp, f) for dp, dn, fn in os.walk(replay_folder) for f in fn if f.endswith(".replay")]

    file_iter = tqdm.tqdm(enumerate(files[:limit]),
                          desc="Load",
                          total=limit,
                          bar_format="{l_bar}{r_bar}")

    datasets = []
    for i, (dp, f) in file_iter:
        try:
            if replay_folder is None:
                out_path = os.path.join(dp, f)
                with open(out_path, "rb") as handle:
                    dfs = pickle.load(handle)
            else:
                in_path = os.path.join(dp, f)
                out_path = os.path.join(cache_folder, f[:-7] + ".pickle")
                if os.path.exists(out_path):
                    with open(out_path, "rb") as handle:
                        dfs = pickle.load(handle)
                else:
                    dfs = replay_to_dfs(in_path)
                    with open(out_path, "wb") as handle:
                        pickle.dump(dfs, handle)

            x_n, y_n = convert_dfs(dfs, tensors=True)
            assert x_n[2].shape == x_n[3].shape
            normalize(x_n)

            swap_teams(x_n, y_n, slice(i % 2, None, 2))

            datasets.append(TensorDataset(*x_n, *y_n))

            # x_s, y_s = [v.copy() for v in x_n], [v.copy() for v in y_n]
            # swap_teams(x_s, y_s)

            # arrays.append((x_s, y_s))
        except Exception as e:
            print(e)
            pass

    ds = ConcatDataset(datasets)
    if name is not None:
        torch.save(ds, name)
    return ds
