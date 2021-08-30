import logging
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from ..util.constants import DEFAULT_FEATURES, IS_BALL, IS_BOOST, BALL_COLS, PLAYER_COLS, DEFAULT_FEATURES_STR, IS_BLUE, \
    IS_ORANGE, POS_X, POS_Y, POS_Z, BOOST_AMOUNT, IS_DEMOED
from ..util.util import boost_locations, rotator_to_matrix

X_DATASET_NAMES = ("balls", "boosts", "players")
Y_DATASET_NAMES = ("next_goal", "next_touch", "next_boost", "next_demo",
                   "control_throttle", "control_steer", "control_pitch", "control_yaw", "control_roll",
                   "control_jump", "control_boost", "control_handbrake")


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
                try:
                    replay_path = os.path.join(replay_folder, replay_file)
                    processed = replay_to_dfs(replay_path)
                    with open(cache_file, "wb") as handle:
                        pickle.dump(processed, handle)
                except Exception as e:
                    # raise e
                    print(e)
                    continue
            yield convert_dfs(processed)
    elif replay_folder is not None and cache_folder is None:
        for replay_file in tqdm(os.listdir(replay_folder), "Replay processing"):
            try:
                replay_path = os.path.join(replay_folder, replay_file)
                processed = replay_to_dfs(replay_path, skip_ties=False, skip_kickoffs=True)
                yield convert_dfs(processed)
            except Exception as e:
                # raise e
                print(e)
                continue
    elif replay_folder is None and cache_folder is not None:
        for cache_file in tqdm(os.listdir(cache_folder), "Replay loading"):
            with open(cache_file, "rb") as handle:
                processed = pickle.load(handle)
            yield convert_dfs(processed)


def replay_to_dfs(replay, skip_ties=True, skip_kickoffs=True):
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
    player_df = pd.DataFrame(columns=["color", "online_id", "name"])
    i = 0
    for color, team in ("blue", blue_team), ("orange", orange_team):
        for player in team.players:
            player_index[player.online_id] = i
            frames[f"{i}/pos_x"] = player.data["pos_x"].fillna(0.)
            frames[f"{i}/pos_y"] = player.data["pos_y"].fillna(0.)
            frames[f"{i}/pos_z"] = player.data["pos_z"].fillna(0.)

            yaw = player.data["rot_y"].fillna(0.)
            pitch = player.data["rot_x"].fillna(0.)
            roll = player.data["rot_z"].fillna(0.)
            forward, up = rotator_to_matrix(yaw, pitch, roll)

            frames[f"{i}/forward_x"] = forward[0]
            frames[f"{i}/forward_y"] = forward[1]
            frames[f"{i}/forward_z"] = forward[2]

            frames[f"{i}/up_x"] = up[0]
            frames[f"{i}/up_y"] = up[1]
            frames[f"{i}/up_z"] = up[2]

            frames[f"{i}/vel_x"] = player.data["vel_x"].fillna(0.) / 10
            frames[f"{i}/vel_y"] = player.data["vel_y"].fillna(0.) / 10
            frames[f"{i}/vel_z"] = player.data["vel_z"].fillna(0.) / 10

            frames[f"{i}/ang_vel_x"] = player.data["ang_vel_x"].fillna(0.) / 1000
            frames[f"{i}/ang_vel_y"] = player.data["ang_vel_y"].fillna(0.) / 1000
            frames[f"{i}/ang_vel_z"] = player.data["ang_vel_z"].fillna(0.) / 1000

            frames[f"{i}/boost_amount"] = player.data["boost"].fillna(0.) / 2.55

            for col in player.controls.columns:
                frames[f"{i}/{col}_control"] = player.controls[col].astype(float)

            boost_pickups = player.data["boost"].diff() > 0
            for frame, row in player.data[boost_pickups].iterrows():
                xyz = (row["pos_x"], row["pos_y"], row["pos_z"])
                # TODO use boost ID instead
                closest_boost = min(range(len(boost_locations)),
                                    key=lambda j: sum((a - b) ** 2 for a, b in zip(xyz, boost_locations[j])))
                boost_grabs.loc[len(boost_grabs)] = [frame, closest_boost, i]

            player_df.loc[len(player_df)] = [color, player.online_id, player.name]
            i += 1

    rallies = pd.DataFrame(columns=["start_frame", "end_frame", "team"])
    for kf1, kf2 in zip(replay.game.kickoff_frames, replay.game.kickoff_frames[1:] + [replay.game.frames.index[-1]]):
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

    if skip_ties:
        goal_rallies = rallies[rallies["team"].notna()]
        frames = frames.loc[
            np.r_[tuple(slice(row["start_frame"], row["end_frame"]) for _, row in goal_rallies.iterrows())]]
    elif skip_kickoffs:
        goal_rallies = rallies
        frames = frames.loc[
            np.r_[tuple(slice(row["start_frame"], row["end_frame"]) for _, row in goal_rallies.iterrows())]]

    # if isinstance(frame_mode, int):
    #     frames = frames.iloc[np.random.randint(0, frame_mode)::frame_mode]
    # else:
    #     frames = frames.sample(frac=frame_mode)

    return {"frames": frames, "rallies": rallies, "touches": touches, "boost_grabs": boost_grabs, "demos": demos,
            "players": player_df}


def convert_dfs(dfs, tensors=False, frame_mode=1):
    frames, rallies, touches, boost_grabs, demos, player_df = \
        (dfs[k] for k in ("frames", "rallies", "touches", "boost_grabs", "demos", "players"))

    control_cols = frames.filter(regex="_control$").columns
    frames[control_cols] = frames[control_cols].shift()
    frames = frames.iloc[np.random.randint(0, frame_mode)::frame_mode]

    if tensors:
        convert_fn = lambda x: torch.from_numpy(x).to(torch.float)
    else:
        convert_fn = lambda x: x

    x, y = get_base_features(len(frames), 6, tensors=tensors)
    x_ball, x_boost, x_players = x
    (y_score, y_next_touch, y_collect, y_demo,
     y_throttle, y_steer, y_pitch, y_yaw, y_roll, y_jump, y_boost, y_handbrake) = y

    x_ball[:, 0, BALL_COLS] = \
        convert_fn(
            frames[[f"ball/{col}" for col in map(DEFAULT_FEATURES_STR.__getitem__, BALL_COLS)]].values)

    bins = np.array([-50, -0.1, 0.1])  # np.arange(0.25 / 2 - 1, 1, 0.25)

    for i, row in player_df.iterrows():
        color = row["color"]
        if color == "blue":
            x_players[:, i, IS_BLUE] = 1
        elif color == "orange":
            x_players[:, i, IS_ORANGE] = 1
        else:
            raise ValueError(color)

        x_players[:, i, PLAYER_COLS[:-3]] = \
            convert_fn(frames[[f"{i}/{col}" for col in
                               map(DEFAULT_FEATURES_STR.__getitem__, PLAYER_COLS[:-3])]].fillna(0.).values)

        throttle = np.digitize(frames[f"{i}/throttle_control"].fillna(-100).values, bins) - 1
        steer = np.digitize(frames[f"{i}/steer_control"].fillna(-100).values, bins) - 1
        pitch = np.digitize(frames[f"{i}/pitch_control"].fillna(-100).values, bins) - 1
        yaw = np.digitize(frames[f"{i}/yaw_control"].fillna(-100).values, bins) - 1
        roll = np.digitize(frames[f"{i}/roll_control"].fillna(-100).values, bins) - 1
        jump = frames[f"{i}/jump_control"].fillna(-100).astype(int).values
        boost = frames[f"{i}/boost_control"].fillna(-100).astype(int).values
        handbrake = frames[f"{i}/handbrake_control"].fillna(-100).astype(int).values

        throttle[throttle < 0] = -100
        steer[steer < 0] = -100
        pitch[pitch < 0] = -100
        yaw[yaw < 0] = -100
        roll[roll < 0] = -100
        jump[jump < 0] = -100
        boost[boost < 0] = -100
        handbrake[handbrake < 0] = -100

        y_throttle[:, i] = convert_fn(throttle)
        y_steer[:, i] = convert_fn(steer)
        y_pitch[:, i] = convert_fn(pitch)
        y_yaw[:, i] = convert_fn(yaw)
        y_roll[:, i] = convert_fn(roll)
        y_jump[:, i] = convert_fn(jump)
        y_boost[:, i] = convert_fn(boost)
        y_handbrake[:, i] = convert_fn(handbrake)

    for i, (x, y, z) in enumerate(boost_locations):
        x_boost[:, i, POS_X] = x
        x_boost[:, i, POS_Y] = y
        x_boost[:, i, POS_Z] = z
        x_boost[:, i, BOOST_AMOUNT] = 100. if z > 72 else 12

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

            indices = (last_frame <= frames.index) & (frames.index < frame)
            y_next_touch[indices, 0] = player
            last_frame = frame

        for i, row in player_df.iterrows():
            player_demos = rally_demos[rally_demos["attacker"] == i]
            player_collects = rally_collects[rally_collects["player"] == i]

            last_frame = rally["start_frame"]
            for _, boost_grab in player_collects.iterrows():
                frame = boost_grab["frame"]
                boost_id = boost_grab["boost_id"]

                if boost_id in (3, 4, 15, 18, 29, 30):
                    collected_indices = (frames.index - frame >= 0) & (frames.index - frame < 10 * 30)
                else:
                    collected_indices = (frames.index - frame >= 0) & (frames.index - frame < 4 * 30)

                x_boost[collected_indices, boost_id, BOOST_AMOUNT] = 0.
                x_boost[collected_indices, boost_id, IS_DEMOED] = 1.

                indices = (last_frame <= frames.index) & (frames.index < frame)
                y_collect[indices, i] = boost_id
                last_frame = frame

            last_frame = rally["start_frame"]
            for _, demo in player_demos.iterrows():
                frame = demo["frame"]

                j = demo["victim"]

                demoed_indices = (frames.index - frame >= 0) & (frames.index - frame < 30 * 3)
                x_players[demoed_indices, j, IS_DEMOED] = 1

                indices = (last_frame <= frames.index) & (frames.index < frame)
                y_demo[indices, i] = j
                last_frame = frame
    x_data = [x_ball, x_boost, x_players]
    y_data = [y_score, y_next_touch, y_collect, y_demo,
              y_throttle, y_steer, y_pitch, y_yaw, y_roll, y_jump, y_boost, y_handbrake]

    # indices = np.concatenate([y_score[20:] - y_score[:-20] != 0, np.array([False] * 20)])
    # x_data = [v[indices] for v in x_data]
    # y_data = [v[indices] for v in y_data]
    return x_data, y_data


def get_base_features(size, n_players, n_balls=1, n_boosts=34, include_y=True, tensors=False, n_features=None):
    if n_features is None:
        n_features = len(DEFAULT_FEATURES)

    if tensors:
        initializer = torch
    else:
        initializer = np

    x_ball = initializer.zeros((size, n_balls, n_features))
    x_boost = initializer.zeros((size, n_boosts, n_features))
    x_players = initializer.zeros((size, n_players, n_features))

    x_ball[:, :, IS_BALL] = 1
    x_boost[:, :, IS_BOOST] = 1

    if not include_y:
        return [x_ball, x_boost, x_players]

    y_score = initializer.full((size,), -100)
    y_next_touch = initializer.full((size, n_balls,), -100)
    y_collect = initializer.full((size, n_players,), -100)
    y_demo = initializer.full((size, n_players,), -100)

    y_throttle = initializer.full((size, n_players), -100)
    y_steer = initializer.full((size, n_players), -100)
    y_pitch = initializer.full((size, n_players), -100)
    y_yaw = initializer.full((size, n_players), -100)
    y_roll = initializer.full((size, n_players), -100)
    y_jump = initializer.full((size, n_players), -100)
    y_boost = initializer.full((size, n_players), -100)
    y_handbrake = initializer.full((size, n_players), -100)

    return [x_ball, x_boost, x_players], \
           [y_score, y_next_touch, y_collect, y_demo,
            y_throttle, y_steer, y_pitch, y_yaw, y_roll, y_jump, y_boost, y_handbrake]


def normalize(x_data):
    norms = np.array([
        1.,
        1., 1., 1., 1.,
        2300., 2300., 2300.,
        1., 1., 1.,
        1., 1., 1.,
        2300., 2300., 2300.,
        5.5, 5.5, 5.5,
        100., 1., 1., 1.
    ], dtype=np.float32)[:x_data[0].shape[-1]]

    for i in range(len(x_data)):
        x_data[i] /= norms


def swap_teams(x_data, y_data=None, indices=None):
    if indices is None:
        indices = slice(None)

    swaps = np.array([
        1.,
        1., 1., 1., 1.,
        -1., -1., 1.,
        -1., -1., 1.,
        -1., -1., 1.,
        -1., -1., 1.,
        -1., -1., 1.,
        1., 1., 1., 1.
    ])[:x_data[0].shape[-1]]

    x_ball, x_boost, x_players = x_data

    x_ball[indices, :, :] *= swaps

    x_boost[indices, :, :] = x_boost[indices, :, :] * swaps

    x_players[indices, :, :] = x_players[indices, :, :] * swaps
    x_players[indices, :, [IS_BLUE, IS_ORANGE]] = x_players[indices, :,
                                                  [IS_ORANGE, IS_BLUE]]  # Swap blue/orange indicator

    if y_data is None:
        return

    (y_score, y_next_touch, y_collect, y_demo,
     y_throttle, y_steer, y_pitch, y_yaw, y_roll, y_jump, y_boost, y_handbrake) = y_data

    # n_players = x_players.shape[1] - 1

    y_score[indices] = 1 - y_score[indices]
    y_score[y_score > 1] = -100

    # y_next_touch[indices, :] = n_players - y_next_touch[indices, :]
    # y_next_touch[y_next_touch > n_players] = -100


def swap_left_right(x_data, y_data=None, indices=None):
    swaps = np.array([
        1.,
        1., 1., 1., 1.,
        -1., 1., 1.,
        -1., 1., 1.,
        -1., 1., 1.,
        -1., 1., 1.,
        -1., 1., 1.,
        1., 1., 1., 1.
    ])[:x_data[0].shape[-1]]
    if indices is None:
        indices = slice(None)
    # Not really necessary for transformer
    x_ball, x_boost, x_players = x_data

    x_ball[indices, :, :] *= swaps
    x_boost[indices, :, :] *= swaps
    x_players[indices, :, :] *= swaps

    if y_data is None:
        return

    (y_score, y_next_touch, y_collect, y_demo,
     y_throttle, y_steer, y_pitch, y_yaw, y_roll, y_jump, y_boost, y_handbrake) = y_data

    y_steer[indices, :] = 2 - y_steer[indices, :]
    y_steer[y_steer > 2] = -100

    y_yaw[indices, :] = 2 - y_yaw[indices, :]
    y_yaw[y_yaw > 2] = -100

    y_roll[indices, :] = 2 - y_roll[indices, :]
    y_roll[y_roll > 2] = -100
