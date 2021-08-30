import logging

import carball as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from carball.controls.controls import ControlsCreator

from ..dataset.create_dataset import replay_to_dfs, convert_dfs, normalize


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inv_sigmoid(x):
    return np.log(x / (1 - x))


def predict_boost_values(rp, model):
    if isinstance(rp, str):
        rp = cb.analyze_replay_file(rp,
                                    controls=ControlsCreator(),
                                    logging_level=logging.CRITICAL)

    pp = replay_to_dfs(rp, skip_ties=False)
    cpx, cpy = convert_dfs(pp, frame_mode=1)
    normalize(cpx)

    results = np.zeros((cpx[0].shape[0], len(pp["players"]), 101))

    # xs = np.zeros((cpx[0].shape[0], len(pp["players"])))
    # ys = np.zeros((cpx[0].shape[0], len(pp["players"])))

    model.eval()
    with torch.no_grad():
        for p, row in pp["players"].iterrows():
            print(p)
            cpxc = [np.copy(v) for v in cpx]

            # # For making histogram
            # preds = model(*[torch.from_numpy(v).float().cuda() for v in cpxc])
            # preds = preds[0].cpu().detach().numpy()
            # preds = preds[:, 1] - preds[:, 0]
            # if row["color"] == "blue":
            #     preds = -preds
            # xs[:, p] = cpxc[2][:, p, 19]
            # ys[:, p] = preds

            # For changing the boost amount at every step
            for b in range(101):
                print("\t", b)
                cpxc[2][:, p, 19] = b / 100
                cpxc[2][:, p, 19][cpxc[2][:, p, 20] == 1] = 0
                preds = model(*[torch.from_numpy(v).float().cuda() for v in cpxc])
                preds = preds[0].cpu().detach().numpy()
                preds = preds[:, 1] - preds[:, 0]
                if row["color"] == "blue":
                    preds = -preds
                results[:, p, b] = preds

    return results


def make_summary(rp, model):
    if isinstance(rp, str):
        rp = cb.analyze_replay_file(rp,
                                    logging_level=logging.CRITICAL)

    pp = replay_to_dfs(rp, skip_ties=False)

    cpx, cpy = convert_dfs(pp, frame_mode=1)
    # print(cpy)

    normalize(cpx)

    pred_df = pd.DataFrame(index=pp["frames"].index)

    model.eval()

    with torch.no_grad():

        def get_predictions(x_data):
            preds = None
            bs = 1024
            lo, hi = 0, bs
            for batch in range(bs, len(x_data[0]) + bs, bs):
                hi = min(batch, len(x_data[0]))

                batch_x = [v[lo:hi] for v in x_data]
                batch_preds_b = model(*[torch.from_numpy(v).float().cuda() for v in batch_x])
                batch_preds_b = [bp.cpu().detach().numpy() for bp in batch_preds_b]

                # swap_teams(batch_x)
                # batch_preds_s = model(*[torch.from_numpy(v).float().cuda() for v in batch_x])
                # batch_preds_s = [bp.cpu().detach().numpy() for bp in batch_preds_s]
                #
                # swap_left_right(batch_x)
                # batch_preds_sm = model(*[torch.from_numpy(v).float().cuda() for v in batch_x])
                # batch_preds_sm = [bp.cpu().detach().numpy() for bp in batch_preds_sm]
                #
                # swap_teams(batch_x)
                # batch_preds_m = model(*[torch.from_numpy(v).float().cuda() for v in batch_x])
                # batch_preds_m = [bp.cpu().detach().numpy() for bp in batch_preds_m]
                #
                # batch_preds = [(b + s + m + sm) / 4 for b, s, m, sm in
                #                zip(batch_preds_b, batch_preds_s, batch_preds_m, batch_preds_sm)]
                batch_preds = batch_preds_b

                if preds is None:
                    preds = batch_preds
                else:
                    preds = [np.concatenate([p, bp]) for p, bp in zip(preds, batch_preds)]
                lo = hi
            return preds

        # 1. Get predictions with all players
        preds = get_predictions(cpx)
        pred_df["pred"] = preds[0][:, 0] - preds[0][:, 1]

        for i, row in pp["players"].iterrows():
            color = row["color"]
            sgn = 1 if color == "blue" else -1
            cpxc = [np.copy(v) for v in cpx]
            cpxc[2] = cpxc[2][:, [k for k in range(cpx[2].shape[1]) if k != i]]

            # 2. Get predictions with one player removed
            preds = get_predictions(cpxc)
            pred_df[f"{i}/pred_removed"] = preds[0][:, 0] - preds[0][:, 1]

            cpxc = [np.copy(v) for v in cpx]
            cpxc[2] = cpxc[2][:, [i]]

            # 3. Get predictions with teammates removed
            preds = get_predictions(cpxc)
            pred_df[f"{i}/pred_solo"] = preds[0][:, 0] - preds[0][:, 1]

            # pred_df[f"{identifier}/pred"] = sigmoid(sgn * pred_df["pred"]) * \
            #                                 sigmoid(sgn * pred_df[f"{identifier}/pred_solo"]) / \
            #                                 sigmoid(sgn * pred_df[f"{identifier}/pred_removed"])

            pred_df[f"{i}/pred"] = sgn * (
                    pred_df["pred"] - pred_df[f"{i}/pred_removed"])

            # pred_df[f"{identifier}/pred"] = sgn * (
            #         pred_df["pred"] - pred_df[f"{identifier}/pred_removed"] + pred_df[f"{identifier}/pred_solo"])

        return pred_df, pp


def plot_replay(rp_path, model, plot_players=True, invert=None, plot_goal_distance=False):
    if plot_players is True:
        plot_players = ("blue", "orange")
    elif plot_players is False:
        plot_players = ()
    elif plot_players == "blue":
        plot_players = ("blue",)
    elif plot_players == "orange":
        plot_players = ("orange",)
    else:
        plot_players = ()

    rp = cb.analyze_replay_file(rp_path,
                                controls=ControlsCreator(),
                                logging_level=logging.CRITICAL)

    id_team = {p.online_id: ["blue", "orange"][p.team.is_orange] for p in rp.game.players}
    id_name = {p.online_id: p.name for p in rp.game.players}

    pred_df, dfs = make_summary(rp, model)

    if invert is None:
        d = 0
        for goal in rp.game.goals:
            if goal.player.is_orange:
                d += 1
            else:
                d -= 1
        invert = d < 0

    # pp["pred"][pp["ball/ball/vel_y"] == 0] = float("nan")
    # pp["pred"][pp["ball/ball/vel_y"].isna()] = float("nan")

    fig, ax = plt.subplots(figsize=((6.4 * dfs["frames"].index[-1] + 1000) / 5000, 4.8), dpi=400,
                           constrained_layout=True)

    ax.hlines(0.5, xmin=0, xmax=dfs["frames"].index[-1], color="black", linestyle="dashed")

    blue_shots = []
    orange_shots = []

    blue_saves = []
    orange_saves = []

    for hit in rp.protobuf_game.game_stats.hits:
        if hit.shot:
            if id_team[hit.player_id.id] == "blue":
                blue_shots.append(hit.frame_number)
            else:
                orange_shots.append(hit.frame_number)
        if hit.save:
            if id_team[hit.player_id.id] == "blue":
                blue_saves.append(hit.frame_number)
            else:
                orange_saves.append(hit.frame_number)

    blue_ymin, blue_ymax, orange_ymin, orange_ymax = (0.5, 1, 0, 0.5) if invert else (0, 0.5, 0.5, 1)

    ax.vlines(blue_shots, ymin=blue_ymin, ymax=blue_ymax,
              color="m", linestyle="dotted", label="Shot")
    ax.vlines(orange_shots, ymin=orange_ymin, ymax=orange_ymax,
              color="m", linestyle="dotted")

    ax.vlines(blue_saves, ymin=orange_ymin, ymax=orange_ymax,
              color="c", linestyle="dotted", label="Save")
    ax.vlines(orange_saves, ymin=blue_ymin, ymax=blue_ymax,
              color="c", linestyle="dotted")

    if plot_goal_distance:
        goal_location = np.array([0, -5120 if invert else 5120, 642 / 2])
        distances = np.linalg.norm(
            np.tile(goal_location, len(dfs["frames"])).reshape((len(dfs["frames"]), 3)) - dfs["frames"].filter(
                regex="ball/pos_").values, axis=-1)
        ax.plot(dfs["frames"].index, distances / 11216, label="Ball-Goal Distance")

    pred = sigmoid(pred_df["pred"])
    ax.plot(dfs["frames"].index, pred if invert else 1 - pred, color="blue", label="Advantage")

    blue_colors = ["springgreen", "mediumseagreen", "green"]
    orange_colors = ["lightcoral", "indianred", "brown"]
    # plt.show()

    b = o = 0
    for i, row in dfs["players"].iterrows():
        name = row["name"]
        color = row["color"]

        avg_score = np.exp(pred_df[f"{i}/pred"].mean())
        print(name, avg_score)

        # pred_df[f"{identifier}/pred"].plot.hist(title=name, bins=100)
        # plt.show()

        pred_col = sigmoid(pred_df[f"{i}/pred"])
        if color == "blue" and "blue" in plot_players:
            ax.plot(dfs["frames"].index, pred_col if invert else 1 - pred_col, color=blue_colors[b],
                    label=name, linewidth=0.5)
            b += 1
        elif color == "orange" and "orange" in plot_players:
            ax.plot(dfs["frames"].index, 1 - pred_col if invert else pred_col, color=orange_colors[o], label=name,
                    linewidth=0.5)
            o += 1

    rallies = dfs["rallies"]
    ax.vlines(rallies[rallies["team"].notna()]["end_frame"], ymin=0, ymax=1, color="red", label="Goal")
    ax.set_xlim(0, dfs["frames"].index[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Orange <-> Blue" if invert else "Blue <-> Orange")
    ax.set_title(rp.game.name)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    secax = ax.twiny()
    new_tick_locations = []
    new_ticks = []
    taken = None
    for frame in reversed(rp.game.frames.index):
        left = rp.game.frames["seconds_remaining"].loc[frame]
        if np.isnan(left):
            continue
        left = int(left)
        if left % 10 == 0 and left != taken:
            new_tick_locations.append(frame)
            new_ticks.append("{0:.0f}:{1:02.0f}".format(*divmod(rp.game.frames["seconds_remaining"].loc[frame], 60)))
            taken = left
    secax.set_xlim(*ax.get_xlim())
    secax.set_xticks(new_tick_locations)
    secax.set_xticklabels(new_ticks)
    secax.set_xlabel("Time")

    ax.grid()
    # df["pred"].plot(x=df["game/info/frame"], color="blue")
    return fig, dfs


if __name__ == '__main__':
    import torch

    mdl = torch.load("../../out/models/EARLReplayModel(EARL(n_dims=256,n_layers=8,n_heads=8))_trained.model.ep3").cuda()
    # mdl = EARLReplayModel().cuda()
    nrg_worlds = r"C:\Users\rolv_\Downloads\012194B14489DD6AD776F0A877C53C05.replay"
    bds = r"C:\Users\rolv_\Downloads\ffaed1b9-ac42-4923-9e5b-66eaeef135c1.replay"
    # eltaco_loss = r"C:\Users\rolv_\Downloads\987fe5bd-4b18-4678-8a4f-afabb443b3f9.replay"
    eltaco_win = r"C:\Users\rolv_\Downloads\62a59623-081f-4caf-8eaa-b9b3b5bbb1fd.replay"

    # xs, ys = predict_boost_values(bds, mdl)
    # df = pd.DataFrame({"x": xs * 100, "y": ys})
    # means = df.groupby(df["x"].round()).mean()
    # means = (means["y"] - means["y"][0.]) / (means["y"][1.] - means["y"][0.])
    # plt.plot(means)
    # plt.show()

    boost_values = predict_boost_values(bds, mdl)
    np.save("boost_values.npy", boost_values)

    # boost_values = np.load("boost_values.npy")
    # boost_values = sigmoid(boost_values)
    boost_values = (boost_values - boost_values[:, :, [0]]) / (boost_values[:, :, [100]] - boost_values[:, :, [0]])
    boost_values = np.nanmean(boost_values, (0, 1), )
    print("{" + ", ".join(f"({b / 100}, {boost_values[b]})" for b in range(101)) + "}")
    plt.plot(boost_values)
    plt.show()

    # f, p = plot_replay(nrg_worlds, mdl)
    # f.show()
