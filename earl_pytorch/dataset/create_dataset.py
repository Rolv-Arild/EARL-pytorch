import logging
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from earl_pytorch.util.util import boost_locations, rotator_to_matrix

FEATURES = [
    "is_ball", "is_boost", "is_blue", "is_orange",
    "pos_x", "pos_y", "pos_z",
    "forward_x", "forward_y", "forward_z",
    "up_x", "up_y", "up_z",
    "vel_x", "vel_y", "vel_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "boost_amount", "is_demoed"
]

LABELS = [
    "score", "next_touch", "boost_collect",
    "demo", "ground_control", "aerial_control"
]

BALL_COLS = [4, 5, 6, 13, 14, 15, 16, 17, 18]
BOOST_COLS = [4, 5, 6, 19, 20]
PLAYER_COLS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def iterate_replays(bc_api, replay_ids=None, replay_folder=None, cache_folder=None):
    if replay_ids is not None:
        assert replay_folder is not None
        import ballchasing as bc
        bc_api: bc.Api
        for replay_id in tqdm(replay_ids, "Replay download"):
            if not os.path.exists(os.path.join(replay_folder, replay_id + ".replay")):
                bc_api.download_replay(replay_id, replay_folder)

    if replay_folder is not None and cache_folder is not None:
        for replay_file in tqdm(os.listdir(replay_folder), "Replay processing"):
            cache_file = os.path.join(cache_folder, replay_file.replace(".replay", ".pickle"))
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as handle:
                    processed = pickle.load(handle)
            else:
                # try
                replay_path = os.path.join(replay_folder, replay_file)
                processed = replay_to_dfs(replay_path)
                processed = convert_dfs(processed)
                with open(cache_file, "wb") as handle:
                    pickle.dump(processed, handle)
            yield processed
    elif replay_folder is not None and cache_folder is None:
        for replay_file in tqdm(os.listdir(replay_folder), "Replay processing"):
            replay_path = os.path.join(replay_folder, replay_file)
            processed = replay_to_dfs(replay_path)
            processed = convert_dfs(processed)
            yield processed
    elif replay_folder is None and cache_folder is not None:
        for cache_file in tqdm(os.listdir(cache_folder), "Replay loading"):
            with open(cache_file, "rb") as handle:
                processed = pickle.load(handle)
            yield processed


def replay_to_dfs(replay):
    import pandas as pd
    import carball as cb

    if isinstance(replay, (str, bytes)):
        from carball.controls.controls import ControlsCreator
        replay = cb.analyze_replay_file(replay, controls=ControlsCreator(), logging_level=logging.CRITICAL)

    blue_team, orange_team = replay.game.teams
    if blue_team.is_orange:
        blue_team, orange_team = orange_team, blue_team

    frames = pd.DataFrame(index=replay.data_frame.index)

    frames[f"ball/pos_x"] = replay.game.ball["pos_x"].fillna(0.)
    frames[f"ball/pos_y"] = replay.game.ball["pos_y"].fillna(0.)
    frames[f"ball/pos_z"] = replay.game.ball["pos_z"].fillna(0.)

    frames[f"ball/vel_x"] = replay.game.ball["vel_x"].fillna(0.) / 10
    frames[f"ball/vel_y"] = replay.game.ball["vel_y"].fillna(0.) / 10
    frames[f"ball/vel_z"] = replay.game.ball["vel_z"].fillna(0.) / 10

    frames[f"ball/ang_vel_x"] = replay.game.ball["ang_vel_x"].fillna(0.) / 1000
    frames[f"ball/ang_vel_y"] = replay.game.ball["ang_vel_y"].fillna(0.) / 1000
    frames[f"ball/ang_vel_z"] = replay.game.ball["ang_vel_z"].fillna(0.) / 1000

    boost_grabs = pd.DataFrame(columns=["frame", "boost_id", "player"])

    player_index = {}
    player_df = pd.DataFrame(columns=["identifier", "online_id", "name"])
    for color, team in ("blue", blue_team), ("orange", orange_team):
        for n, player in enumerate(team.players):
            identifier = f"{color}_{n}"
            player_index[player.online_id] = identifier
            frames[f"{identifier}/pos_x"] = player.data["pos_x"].fillna(0.)
            frames[f"{identifier}/pos_y"] = player.data["pos_y"].fillna(0.)
            frames[f"{identifier}/pos_z"] = player.data["pos_z"].fillna(0.)

            yaw = player.data["rot_y"].fillna(0.)
            pitch = player.data["rot_x"].fillna(0.)
            roll = player.data["rot_z"].fillna(0.)
            forward, up = rotator_to_matrix(yaw, pitch, roll)

            frames[f"{identifier}/forward_x"] = forward[0]
            frames[f"{identifier}/forward_y"] = forward[1]
            frames[f"{identifier}/forward_z"] = forward[2]

            frames[f"{identifier}/up_x"] = up[0]
            frames[f"{identifier}/up_y"] = up[1]
            frames[f"{identifier}/up_z"] = up[2]

            frames[f"{identifier}/vel_x"] = player.data["vel_x"].fillna(0.) / 10
            frames[f"{identifier}/vel_y"] = player.data["vel_y"].fillna(0.) / 10
            frames[f"{identifier}/vel_z"] = player.data["vel_z"].fillna(0.) / 10

            frames[f"{identifier}/ang_vel_x"] = player.data["ang_vel_x"].fillna(0.) / 1000
            frames[f"{identifier}/ang_vel_y"] = player.data["ang_vel_y"].fillna(0.) / 1000
            frames[f"{identifier}/ang_vel_z"] = player.data["ang_vel_z"].fillna(0.) / 1000

            frames[f"{identifier}/boost_amount"] = player.data["boost"].fillna(0.) / 2.55

            for col in player.controls.columns:
                frames[f"{identifier}/{col}_control"] = player.controls[col].astype(float).fillna(0.)

            boost_pickups = player.data["boost"].diff() > 0
            for frame, row in player.data[boost_pickups].iterrows():
                xyz = (row["pos_x"], row["pos_y"], row["pos_z"])
                # TODO use boost ID instead
                closest_boost = min(range(len(boost_locations)),
                                    key=lambda i: sum((a - b) ** 2 for a, b in zip(xyz, boost_locations[i])))
                boost_grabs.loc[len(boost_grabs)] = [frame, closest_boost, identifier]

            player_df.loc[len(player_df)] = [identifier, player.online_id, player.name]

    rallies = pd.DataFrame(columns=["start_frame", "end_frame", "team"])
    for kf1, kf2 in zip(replay.game.kickoff_frames, replay.game.kickoff_frames[1:] + [float("inf")]):
        for goal in replay.game.goals:
            if kf1 < goal.frame_number < kf2:
                rallies.loc[len(rallies)] = [kf1, goal.frame_number, ["blue", "orange"][goal.player_team]]
                break
        else:  # No goal between kickoffs
            rallies.loc[len(rallies)] = [kf1, kf2, None]

    demos = pd.DataFrame(columns=["frame", "attacker", "victim"])
    for demo in replay.game.demos:
        frame = demo.get("frame_number", None)
        attacker = demo.get("attacker", None)
        victim = demo.get("victim", None)

        if frame and attacker and victim:
            demos.loc[len(demos)] = [frame, player_index[attacker.online_id], player_index[victim.online_id]]

    touches = pd.DataFrame(columns=["frame", "player"])
    for touch in replay.protobuf_game.game_stats.hits:
        touches.loc[len(touches)] = [touch.frame_number, player_index[touch.player_id.id]]

    goal_rallies = rallies[rallies["team"].notna()]
    frames = frames.loc[
        np.r_[tuple(slice(row["start_frame"], row["end_frame"]) for _, row in goal_rallies.iterrows())]]
    frames = frames.iloc[np.random.randint(0, 15)::15]

    return {"frames": frames, "rallies": rallies, "touches": touches, "boost_grabs": boost_grabs, "demos": demos,
            "players": player_df}


def convert_dfs(dfs, tensors=False):
    frames, rallies, touches, boost_grabs, demos, player_df = \
        (dfs[k] for k in ("frames", "rallies", "touches", "boost_grabs", "demos", "players"))

    n_blue = (player_df["identifier"].str.contains("blue")).sum()
    n_orange = (player_df["identifier"].str.contains("orange")).sum()
    x, y = get_base_features(len(frames), n_blue, n_orange, tensors=tensors)
    if tensors:
        convert_fn = lambda x: torch.from_numpy(x).to(torch.float)
    else:
        convert_fn = lambda x: x

    x_ball, x_boost, x_blue, x_orange = x
    y_score, y_next_touch, y_collect, y_demo, y_ground_control, y_aerial_control = y

    x_ball[:, 0, BALL_COLS] = \
        convert_fn(
            frames[[f"ball/{col}" for col in map(FEATURES.__getitem__, BALL_COLS)]].values)

    bins = np.array([-0.1, 0.1])  # np.arange(0.25 / 2 - 1, 1, 0.25)

    for identifier in player_df["identifier"]:
        color, n = identifier.split("_")
        n = int(n)
        i = n + n_blue * (color == "orange")
        if color == "blue":
            team = x_blue
        elif color == "orange":
            team = x_orange
        else:
            raise ValueError(color)

        team[:, n, PLAYER_COLS] = \
            convert_fn(frames[[f"{identifier}/{col}" for col in
                               map(FEATURES.__getitem__, PLAYER_COLS)]].values)

        throttle = np.digitize(frames[f"{identifier}/throttle_control"].values, bins)
        steer = np.digitize(frames[f"{identifier}/steer_control"].values, bins)
        pitch = np.digitize(frames[f"{identifier}/pitch_control"].values, bins)
        yaw = np.digitize(frames[f"{identifier}/yaw_control"].values, bins)
        roll = np.digitize(frames[f"{identifier}/roll_control"].values, bins)
        jump = frames[f"{identifier}/jump_control"].astype(int).values
        boost = frames[f"{identifier}/boost_control"].astype(int).values
        handbrake = frames[f"{identifier}/handbrake_control"].astype(int).values

        y_ground_control[:, i] = convert_fn(throttle + 3 * (steer + 3 * (jump + 2 * (boost + 2 * handbrake))))
        y_aerial_control[:, i] = convert_fn(pitch + 3 * (yaw + 3 * (roll + 3 * (jump + 2 * boost))))

    for i, (x, y, z) in enumerate(boost_locations):
        x_boost[:, i, 4] = x
        x_boost[:, i, 5] = y
        x_boost[:, i, 6] = z
        x_boost[:, i, 19] = 100. if z > 72 else 12

    for _, rally in rallies.iterrows():
        indices = (rally["start_frame"] <= frames.index) & (frames.index <= rally["end_frame"])
        team = rally["team"]
        if team == "blue":
            y_score[indices] = 0
        elif team == "orange":
            y_score[indices] = 1
        else:
            # print("Team:", team)
            pass

        rally_touches = touches[
            (rally["start_frame"] <= touches["frame"]) & (touches["frame"] < rally["end_frame"])]
        rally_demos = demos[(rally["start_frame"] <= demos["frame"]) & (demos["frame"] < rally["end_frame"])]
        rally_collects = boost_grabs[
            (rally["start_frame"] <= boost_grabs["frame"]) & (boost_grabs["frame"] < rally["end_frame"])]

        last_frame = rally["start_frame"]
        for _, touch in rally_touches.iterrows():
            frame = touch["frame"]
            player = touch["player"]
            color, i = player.split("_")
            i = int(i) + n_blue * (color == "orange")

            indices = (last_frame <= frames.index) & (frames.index < frame)
            y_next_touch[indices, 0] = i
            last_frame = frame

        for identifier in player_df["identifier"]:
            player_demos = rally_demos[rally_demos["attacker"] == identifier]
            player_collects = rally_collects[rally_collects["player"] == identifier]

            color, n = identifier.split("_")
            n = int(n)
            i = n + n_blue * (color == "orange")

            last_frame = rally["start_frame"]
            for _, boost_grab in player_collects.iterrows():
                frame = boost_grab["frame"]
                boost_id = boost_grab["boost_id"]

                if boost_id in (3, 4, 15, 18, 29, 30):
                    collected_indices = (frames.index - frame >= 0) & (frames.index - frame < 10 * 30)
                else:
                    collected_indices = (frames.index - frame >= 0) & (frames.index - frame < 4 * 30)

                x_boost[collected_indices, boost_id, 19] = 0.
                x_boost[collected_indices, boost_id, 20] = 1.

                indices = (last_frame <= frames.index) & (frames.index < frame)
                y_collect[indices, i] = boost_id
                last_frame = frame

            last_frame = rally["start_frame"]
            for _, demo in player_demos.iterrows():
                frame = demo["frame"]

                victim = demo["victim"]
                v_color, j = victim.split("_")
                j = int(j)

                demoed_indices = (frames.index - frame >= 0) & (frames.index - frame < 30 * 3)
                if v_color == "blue":
                    x_blue[demoed_indices, j, 20] = 1
                elif v_color == "orange":
                    x_orange[demoed_indices, j, 20] = 1
                else:
                    print("Team:", v_color)

                indices = (last_frame <= frames.index) & (frames.index < frame)
                y_demo[indices, i] = j + n_blue * (v_color == "orange")
                last_frame = frame
    x_data = [x_ball, x_boost, x_blue, x_orange]
    y_data = [y_score, y_next_touch, y_collect, y_demo, y_ground_control, y_aerial_control]

    # indices = np.concatenate([y_score[20:] - y_score[:-20] != 0, np.array([False] * 20)])
    # x_data = [v[indices] for v in x_data]
    # y_data = [v[indices] for v in y_data]
    return x_data, y_data


def get_base_features(size, n_blue, n_orange, n_balls=1, n_boosts=34, include_y=True, tensors=False):
    n_features = len(FEATURES)
    if tensors:
        initializer = torch
    else:
        initializer = np

    x_ball = initializer.zeros((size, n_balls, n_features))
    x_boost = initializer.zeros((size, n_boosts, n_features))
    x_blue = initializer.zeros((size, n_blue, n_features))
    x_orange = initializer.zeros((size, n_orange, n_features))

    x_ball[:, :, 0] = 1
    x_boost[:, :, 1] = 1
    x_blue[:, :, 2] = 1
    x_orange[:, :, 3] = 1

    if not include_y:
        return [x_ball, x_boost, x_blue, x_orange]

    y_score = initializer.full((size,), -100)
    y_next_touch = initializer.full((size, n_balls,), -100)
    y_collect = initializer.full((size, n_blue + n_orange,), -100)
    y_demo = initializer.full((size, n_blue + n_orange,), -100)
    y_ground_control = initializer.full((size, n_blue + n_orange), -100)
    y_aerial_control = initializer.full((size, n_blue + n_orange), -100)

    return [x_ball, x_boost, x_blue, x_orange], \
           [y_score, y_next_touch, y_collect, y_demo, y_ground_control, y_aerial_control]


def normalize(x_data):
    norms = np.array([
        1., 1., 1., 1.,
        4096., 5120., 2044.,
        1., 1., 1.,
        1., 1., 1.,
        2300., 2300., 2300.,
        5.5, 5.5, 5.5,
        100., 1.
    ])
    x_ball, x_boost, x_blue, x_orange = x_data

    x_ball /= norms
    x_boost /= norms
    x_blue /= norms
    x_orange /= norms


def swap_teams(x_data, y_data=None, indices=None):
    if indices is None:
        indices = slice(None)

    swaps = np.array([
        1., 1., 1., 1.,
        -1., -1., 1.,
        -1., -1., 1.,
        -1., -1., 1.,
        -1., -1., 1.,
        -1., -1., 1.,
        1., 1.
    ])
    x_ball, x_boost, x_blue, x_orange = x_data

    x_ball[indices, :, :] *= swaps
    if isinstance(x_boost, torch.Tensor):
        x_boost[indices, :, :] = x_boost[indices, :, :].flip(1) * swaps
        x_blue[indices, :, 4:], x_orange[indices, :, 4:] = \
            x_orange[indices, :, 4:].flip(1) * swaps[4:], x_blue[indices, :, 4:].flip(1) * swaps[4:]
    else:
        x_boost[indices, :, :] = x_boost[indices, ::-1, :] * swaps
        x_blue[indices, :, 4:], x_orange[indices, :, 4:] = \
            x_orange[indices, ::-1, 4:] * swaps[4:], x_blue[indices, ::-1, 4:] * swaps[4:]

    if y_data is None:
        return

    y_score, y_next_touch, y_collect, y_demo, y_ground_control, y_aerial_control = y_data

    n_players = x_blue.shape[1] + x_orange.shape[1] - 1
    n_boosts = x_boost.shape[1] - 1

    y_score[indices] = 1 - y_score[indices]
    y_score[y_score > 1] = -100

    y_next_touch[indices, :] = n_players - y_next_touch[indices, :]
    y_next_touch[y_next_touch > n_players] = -100

    if isinstance(y_collect, torch.Tensor):
        y_collect[indices, :] = n_boosts - y_collect[indices, :].flip(1)
        y_collect[y_collect > n_boosts] = -100

        y_demo[indices, :] = n_players - y_demo[indices, :].flip(1)
        y_demo[y_demo > n_players] = -100

        y_ground_control[indices, :] = y_ground_control[indices, :].flip(1)
        y_aerial_control[indices, :] = y_aerial_control[indices, :].flip(1)
    else:
        y_collect[indices, :] = n_boosts - y_collect[indices, ::-1]
        y_collect[y_collect > n_boosts] = -100

        y_demo[indices, :] = n_players - y_demo[indices, ::-1]
        y_demo[y_demo > n_players] = -100

        y_ground_control[indices, :] = y_ground_control[indices, ::-1]
        y_aerial_control[indices, :] = y_aerial_control[indices, ::-1]


def swap_left_right(x_data, y_data=None, indices=None, suppress_warning=False):
    swaps = np.array([
        1., 1., 1., 1.,
        -1., 1., 1.,
        -1., 1., 1.,
        -1., 1., 1.,
        -1., 1., 1.,
        -1., 1., 1.,
        1., 1.
    ])
    # Not really necessary for transformer
    # boost_indices = [0, 2, 1, 4, 3, 6, 5, 7, 9, 8, 11, 10, 14, 13, 12, 18, 17, 16, 15, 21, 20, 19, 23, 22, 25, 24, 28,
    #                  27, 30, 29, 32, 31, 33]
    x_ball, x_boost, x_blue, x_orange = x_data
    x_ball[indices, :, :] *= swaps
    x_boost[indices, :, :] *= swaps
    x_blue[indices, :, :] *= swaps
    x_orange[indices, :, :] *= swaps

    if y_data is None:
        return

    y_score, y_next_touch, y_collect, y_demo, y_ground_control, y_aerial_control = y_data
    y_collect[indices, :] = y_collect[indices, :]

    if not suppress_warning:
        print("TODO: controls were not swapped")  # TODO


if __name__ == '__main__':
    import sys
    import ballchasing as bc

    key = sys.argv[1]

    api = bc.Api(key)
    size = 1_000_000
    buf_x, buf_y = get_base_features(size, 3, 3)
    print((sum(v.nbytes for v in buf_x) + sum(v.nbytes for v in buf_y)) / 1e9)
    i = j = 0
    n = 0
    for replay in iterate_replays(api,
                                  replay_folder=r"D:\rokutleg\replays\rlcsx",
                                  cache_folder=r"D:\rokutleg\processed\rlcsx"):
        x, y = replay

        x_s, y_s = x.copy(), y.copy()
        swap_teams(x_s, y_s)

        x_m, y_m = x.copy(), y.copy()
        swap_left_right(x_m, y_m, suppress_warning=True)

        x_sm, y_sm = x_s.copy(), y_s.copy()
        swap_left_right(x_sm, y_sm, suppress_warning=True)

        if i + 4 * len(x[0]) > len(buf_x[0]):
            np.savez(rf"D:\rokutleg\datasets\x_data-{n}.npz", *[v[:i] for v in buf_x])
            np.savez(rf"D:\rokutleg\datasets\y_data-{n}.npz", *[v[:i] for v in buf_y])
            buf_x, buf_y = get_base_features(size, 3, 3)
            n += 1
            i = j = 0

        for buf_v, v, v_s, v_m, v_sm in zip(buf_x, x, x_s, x_m, x_sm):
            j = i
            buf_v[j:j + len(v)] = v
            j += len(v)
            buf_v[j:j + len(v)] = v_s
            j += len(v_s)
            buf_v[j:j + len(v)] = v_m
            j += len(v_m)
            buf_v[j:j + len(v)] = v_sm
            j += len(v_sm)

        for buf_v, v, v_s, v_m, v_sm in zip(buf_y, y, y_s, y_m, y_sm):
            j = i
            buf_v[j:j + len(v)] = v
            j += len(v)
            buf_v[j:j + len(v)] = v_s
            j += len(v_s)
            buf_v[j:j + len(v)] = v_m
            j += len(v_m)
            buf_v[j:j + len(v)] = v_sm
            j += len(v_sm)

        i = j
    np.savez(rf"D:\rokutleg\datasets\x_data-{n}.npz", *[v[:i] for v in buf_x])
    np.savez(rf"D:\rokutleg\datasets\y_data-{n}.npz", *[v[:i] for v in buf_y])
