import json
import os
import subprocess
import sys
import time
from typing import Tuple, Iterator

import ballchasing
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data.dataset import T_co, IterableDataset
from tqdm.contrib.concurrent import process_map

from earl_pytorch import EARL
from earl_pytorch.dataset.create_dataset import get_base_features, normalize, swap_teams, swap_left_right
from earl_pytorch.util.constants import POS_X, BALL_COLS, DEFAULT_FEATURES_STR, PLAYER_COLS, IS_BLUE, IS_ORANGE, POS_Y, \
    POS_Z, BOOST_AMOUNT, IS_DEMOED, CLS
from earl_pytorch.util.util import rotator_to_matrix, NGPModel
from rlgym.utils.common_values import BOOST_LOCATIONS
from rlgym.utils.gamestates import GameState

command = r'carball.exe -i "{}" -o "{}" parquet'

ENV = os.environ.copy()
ENV["NO_COLOR"] = "1"

boost_locations = np.array(BOOST_LOCATIONS)


def euler_to_quaternion(pitch, yaw, roll):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z


def process_replay(replay_path, output_folder):
    folder, fn = os.path.split(replay_path)
    replay_name = fn.replace(".replay", "")
    processed_folder = os.path.join(output_folder, replay_name)
    os.makedirs(processed_folder, exist_ok=True)

    with open(os.path.join(processed_folder, "carball.o.log"), "w", encoding="utf8") as stdout_f:
        with open(os.path.join(processed_folder, "carball.e.log"), "w", encoding="utf8") as stderr_f:
            return subprocess.run(
                command.format(replay_path, processed_folder),
                stdout=stdout_f,
                stderr=stderr_f,
                env=ENV
            )


class CarballAnalysis:
    METADATA_FNAME = "metadata.json"
    ANALYZER_FNAME = "analyzer.json"
    BALL_FNAME = "__ball.parquet"
    GAME_FNAME = "__game.parquet"
    PLAYER_FNAME = "player_{}.parquet"

    def __init__(self, processed_folder: str):
        # print(processed_folder, self.METADATA_FNAME)
        self.metadata = json.load(open(os.path.join(processed_folder, self.METADATA_FNAME)))
        self.analyzer = json.load(open(os.path.join(processed_folder, self.ANALYZER_FNAME)))

        self.ball = pd.read_parquet(os.path.join(processed_folder, self.BALL_FNAME))
        self.game = pd.read_parquet(os.path.join(processed_folder, self.GAME_FNAME))
        self.players = {}
        for player in self.metadata["players"]:
            uid = player["unique_id"]
            player_path = os.path.join(processed_folder, self.PLAYER_FNAME.format(uid))
            if os.path.exists(player_path):
                self.players[uid] = pd.read_parquet(player_path)

    def to_rlgym(self) -> Iterator[Tuple[GameState, np.ndarray]]:
        ball = self.ball.copy()
        # ball[["quat_w", "quat_x", "quat_y", "quat_z"]] = \
        #     np.asarray(euler_to_quaternion(ball["rot_pitch"], ball["rot_yaw"], ball["rot_roll"])).T
        game = self.game.copy()
        players = {}
        for uid, player_df in self.players.items():
            player_df = player_df.copy()
            # player_df[["quat_w", "quat_x", "quat_y", "quat_z"]] = \
            #     np.asarray(euler_to_quaternion(player_df["rot_pitch"], player_df["rot_yaw"], player_df["rot_roll"])).T
            players[uid] = player_df

        df = pd.DataFrame(index=game.index)
        df.loc[:, "ticks_since_last_transmit"] = (game["delta"].values * 120).round()
        df.loc[:, ["blue_score", "orange_score"]] = 0
        df.loc[:, [f"pad_{n}" for n in range(34)]] = 0

        physics_cols = ["pos_x", "pos_y", "pos_z",
                        "quat_w", "quat_x", "quat_y", "quat_z",
                        "vel_x", "vel_y", "vel_z",
                        "ang_vel_x", "ang_vel_y", "ang_vel_z"]
        invert = np.array([1, -1, 1,
                           1, -1, 1,
                           1, -1, 1])
        ball_physics_cols = [col for col in physics_cols if not col.startswith("quat")]
        df.loc[:, [f"ball/{col}" for col in ball_physics_cols]] = ball[ball_physics_cols].values
        df.loc[:, [f"inverted_ball/{col}" for col in ball_physics_cols]] = ball[ball_physics_cols].values * invert

        controls_df = pd.DataFrame(index=df.index)
        for uid, player_df in players.items():
            df.loc[:, f"{uid}/car_id"] = int(uid)
            df.loc[:, f"{uid}/team_num"] = next(
                int(p["is_orange"]) for p in self.metadata["players"] if p["unique_id"] == uid)
            df.loc[:, [f"{uid}/{col}" for col in physics_cols]] = player_df[physics_cols].values
            df.loc[:, [f"inverted_{uid}/{col}" for col in physics_cols]] = 0  # Get columns in right order first
            df.loc[:, [f"inverted_{uid}/{col}" for col in ball_physics_cols]] = player_df[ball_physics_cols].values \
                                                                                * invert
            df[[f"inverted_{uid}/quat_{axis}" for axis in "wxyz"]] = np.asarray(
                euler_to_quaternion(player_df["rot_pitch"], -player_df["rot_yaw"], player_df["rot_roll"])).T

            df.loc[:, f"{uid}/match_goals"] = player_df["match_goals"]
            df.loc[:, f"{uid}/match_saves"] = player_df["match_saves"]
            df.loc[:, f"{uid}/match_shots"] = player_df["match_shots"]
            df.loc[:, f"{uid}/match_demos"] = 0
            df.loc[:, f"{uid}/match_pickups"] = 0
            df.loc[:, f"{uid}/is_demoed"] = 0
            df.loc[:, f"{uid}/on_ground"] = False
            df.loc[:, f"{uid}/ball_touched"] = False
            df.loc[:, f"{uid}/has_flip"] = False
            df.loc[:, f"{uid}/boost_amount"] = player_df["boost_amount"] / 100

            controls_df.loc[:, f"{uid}/throttle"] = player_df["throttle"] / 127.5 - 1
            controls_df.loc[:, f"{uid}/steer"] = player_df["steer"] / 127.5 - 1
            controls_df.loc[:, f"{uid}/yaw"] = 0
            controls_df.loc[:, f"{uid}/pitch"] = 0
            controls_df.loc[:, f"{uid}/roll"] = 0
            controls_df.loc[:, f"{uid}/jump"] = 0
            controls_df.loc[:, f"{uid}/boost"] = player_df["boost_is_active"]
            controls_df.loc[:, f"{uid}/handbrake"] = player_df["handbrake"]

            for frame, pos in player_df[player_df["boost_pickup"] > 0][["pos_x", "pos_y", "pos_z"]].iterrows():
                boost_id = np.linalg.norm(boost_locations - pos.values, axis=-1).argmin()
                if boost_locations[boost_id][2] > 72:  # Big boost
                    df.loc[frame: frame + 30 * 10, f"pad_{boost_id}"] = 1
                else:  # Small boost
                    df.loc[frame: frame + 30 * 4, f"pad_{boost_id}"] = 1
                df.loc[frame:, f"{uid}/match_pickups"] += 1

        for goal in self.metadata["game"]["goals"]:
            if goal["is_orange"]:
                df.loc[goal["frame"]:, "orange_score"] += 1
            else:
                df.loc[goal["frame"]:, "blue_score"] += 1
            player_id = next(p["unique_id"] for p in self.metadata["players"] if p["name"] == goal["player_name"])
            df.loc[goal["frame"]:, f"{player_id}/match_goals"] += 1

        for demo in self.metadata["demos"]:
            frame = demo["frame_number"]
            attacker = demo["attacker_unique_id"]
            victim = demo["victim_unique_id"]
            df.loc[frame:, f"{attacker}/match_demos"] += 1
            df.loc[frame: frame + 30 * 3, f"{victim}/is_demoed"] = 1

        for hit in self.analyzer["hits"]:
            frame = hit["frame_number"]
            player_id = hit["player_unique_id"]
            df.loc[frame, f"{player_id}/ball_touched"] = 1

        for gameplay_period in self.analyzer["gameplay_periods"]:
            start_frame = gameplay_period["start_frame"]
            end_frame = gameplay_period["goal_frame"]
            if end_frame is None:
                end_frame = df.index[-2]

            yield from zip(
                df.loc[start_frame:end_frame].astype(float).fillna(0).apply(lambda x: GameState(list(x)), axis=1),
                controls_df[start_frame + 1:end_frame + 1]  # Actions are taken at t+1
                    .astype(float).fillna(0).apply(lambda x: np.reshape(x.values, (-1, 8)), axis=1)
            )

    def get_earl_compatible(self, n_players=None):
        if n_players is None:
            n_players = len(self.players)
        x, y = get_base_features(len(self.game), n_players)
        x_ball, x_boost, x_players = x
        y_score, y_next_touch, y_collect, y_demo, \
        y_throttle, y_steer, y_pitch, y_yaw, y_roll, y_jump, y_boost, y_handbrake = y

        player_team = {
            uid: next(int(p["is_orange"]) for p in self.metadata["players"] if p["unique_id"] == uid)
            for uid in self.players
        }

        x_boost[:, :, (POS_X, POS_Y, POS_Z)] = boost_locations
        x_boost[:, :, BOOST_AMOUNT] = 100

        valid_frames = []
        for gameplay_period in self.analyzer["gameplay_periods"]:
            start_frame = gameplay_period["start_frame"]
            end_frame = gameplay_period["end_frame"]
            goal_frame = gameplay_period["goal_frame"]
            if goal_frame is not None:
                end_frame = goal_frame  # Skip goal explosion
                goal = next(g for g in self.metadata["game"]["goals"] if g["frame"] == goal_frame)
                y_score[start_frame: end_frame] = int(goal["is_orange"])
                valid_frames.append(slice(start_frame, end_frame))

            ball_df = self.ball.iloc[start_frame: end_frame]
            x_ball[start_frame:end_frame, 0, BALL_COLS] = ball_df[[DEFAULT_FEATURES_STR[c] for c in BALL_COLS]]

            for p, (uid, player_df) in enumerate(self.players.items()):
                player_df = player_df.iloc[start_frame: end_frame].copy()
                forward, up = rotator_to_matrix(player_df["rot_yaw"], player_df["rot_pitch"], player_df["rot_roll"])
                player_df[['forward_x', 'forward_y', 'forward_z']] = np.array(forward).T
                player_df[['up_x', 'up_y', 'up_z']] = np.array(forward).T
                if player_team[uid] == 0:
                    x_players[start_frame:end_frame, p, IS_BLUE] = 1
                else:
                    assert player_team[uid] == 1
                    x_players[start_frame:end_frame, p, IS_ORANGE] = 1
                x_players[start_frame:end_frame, p, PLAYER_COLS[:-3]] = \
                    player_df[[DEFAULT_FEATURES_STR[c] for c in PLAYER_COLS[:-3]]]

                for frame, pos in player_df[player_df["boost_pickup"] > 0][["pos_x", "pos_y", "pos_z"]].iterrows():
                    boost_id = np.linalg.norm(boost_locations - pos.values, axis=-1).argmin()
                    if boost_locations[boost_id][2] > 72:  # Big boost
                        x_boost[frame: frame + 30 * 10, boost_id, IS_DEMOED] = 1
                    else:  # Small boost
                        x_boost[frame: frame + 30 * 4, boost_id, IS_DEMOED] = 1

        for demo in self.metadata["demos"]:
            frame = demo["frame_number"]
            attacker = demo["attacker_unique_id"]
            victim = demo["victim_unique_id"]

            victim_index = list(self.players).index(victim)

            x_players[frame: frame + 30 * 3, victim_index, IS_DEMOED] = 1

        normalize(x)

        x_s, y_s = [v.copy() for v in x], [v.copy() for v in y]
        swap_teams(x_s, y_s)

        x_m, y_m = [v.copy() for v in x], [v.copy() for v in y]
        swap_left_right(x_m, y_m)

        x_sm, y_sm = [v.copy() for v in x_s], [v.copy() for v in y_s]
        swap_left_right(x_sm, y_sm)

        all_x = []
        all_y = []
        for x_variant, y_variant in ((x, y), (x_s, y_s), (x_m, y_m), (x_sm, y_sm)):
            cls = np.zeros((x_variant[0].shape[0], 1, x_variant[0].shape[2]))
            cls[:, 0, CLS] = 1
            x_variant = np.concatenate(
                [cls] + x_variant,
                axis=1
            )
            y_score = y_variant[0]

            x_variant = x_variant[np.r_[tuple(valid_frames)]]
            y_score = y_score[np.r_[tuple(valid_frames)]]

            np.nan_to_num(x_variant, copy=False)

            all_x.append(x_variant)
            all_y.append(y_score)

        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return all_x, all_y


class ReplayDataset(IterableDataset):
    def __init__(self, folder):
        self.folder = folder

    def __iter__(self) -> Iterator[T_co]:
        for replay_folder in os.listdir(self.folder):
            path = os.path.join(self.folder, replay_folder)
            if os.path.isdir(path):
                ca = CarballAnalysis(path)
                x, y = ca.get_earl_compatible(n_players=6)
                yield x, y

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError


def main():
    group = "rlcs-x-whqc4pi1kw"
    replay_folder = r"D:\rokutleg\replays\rlcsx"
    output_folder = r"D:\rokutleg\processed\rlcsx"
    os.makedirs(replay_folder, exist_ok=True)

    if len(sys.argv) > 1:
        key = sys.argv[1]
        bc_api = ballchasing.Api(key)
        folder = os.path.join(replay_folder, group)
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(replay_folder, "replay_info.ijson"), "w") as info:
            for replay in bc_api.get_group_replays(group):
                try:
                    bc_api.download_replay(replay["id"], folder)
                    info.write(json.dumps(replay) + "\n")
                except Exception as e:
                    print(e)

    all_replays = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(replay_folder)
        for f in fn
        if f.endswith(".replay")
    ]

    process_map(process_replay, all_replays, [output_folder] * len(all_replays), chunksize=5)

    replay_folder = r"D:\rokutleg\replays\electrum"
    output_folder = r"D:\rokutleg\processed\electrum"
    all_replays = [os.path.join(dp, f) for dp, dn, fn in os.walk(replay_folder) for f in fn]

    process_map(process_replay, all_replays, [output_folder] * len(all_replays), chunksize=5)


def earlify(path):
    if os.path.isdir(path):
        try:
            ca = CarballAnalysis(path)
            x, y = ca.get_earl_compatible(n_players=6)
            return x, y
        except Exception as e:
            print(e)
            pass
        # except FileNotFoundError:
        #     print("Not found error when trying to open:", path)
        # except UnicodeDecodeError:
        #     print("Unicode decode error when trying to open:", path)


def data_gen():
    folder = r"D:\rokutleg\processed\rlcsx"
    files = os.listdir(folder)
    # random.shuffle(files)
    files = [os.path.join(folder, replay_folder) for replay_folder in files]

    # with ProcessPoolExecutor(1) as ex:
    for res in map(earlify, files):
        if res is not None:
            yield res


def train():
    model = NGPModel(EARL(dropout=0.1)).cuda()

    bs = 256
    optimizer = Adam(model.parameters(), lr=3e-4)
    loss_fn = CrossEntropyLoss()
    for epoch in range(1, 100):
        n = 0
        xs = []
        ys = []
        torch.save(model, f"EARL-NGP-{epoch}")
        tot_loss = 0
        b = 0
        t = 0
        for x, y in data_gen():
            xs.append(x)
            ys.append(y)

            if n % 8 == 7:
                xs = torch.from_numpy(np.concatenate(xs)).float().cuda()
                ys = torch.from_numpy(np.concatenate(ys)).long().cuda()

                indices = torch.randperm(xs.shape[0])
                xs = xs[indices]
                ys = ys[indices]

                testing = t % 8 == 0
                # tot_loss = 0
                # b = 0
                for batch in range(0, xs.shape[0] - bs, bs):
                    x_batch = xs[batch: batch + bs]
                    y_batch = ys[batch: batch + bs]

                    if not testing:
                        model.train()
                        optimizer.zero_grad()
                        pred = model(x_batch)
                        loss = loss_fn(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                    else:
                        model.eval()
                        with torch.no_grad():
                            pred = model(x_batch)
                            loss = loss_fn(pred, y_batch)
                        tot_loss += loss.item()
                        b += 1

                if testing:
                    print(tot_loss / (b or 1))
                xs = []
                ys = []
                t += 1

            n += 1
        print(epoch, tot_loss / (b or 1))


if __name__ == '__main__':
    # train()
    main()
    exit(0)
    t0 = time.time()
    ca = CarballAnalysis(r"D:\rokutleg\processed\rlcsx\00b8df8c-0088-4560-91b0-a9b9586cc183")
    for frame in ca.to_rlgym():
        pass
    t1 = time.time()
    print("Hei", t1 - t0)
