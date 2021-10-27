import numpy as np

DEFAULT_FEATURES = (
    CLS,
    IS_BALL, IS_BOOST, IS_BLUE, IS_ORANGE,
    POS_X, POS_Y, POS_Z,
    FORWARD_X, FORWARD_Y, FORWARD_Z,
    UP_X, UP_Y, UP_Z,
    VEL_X, VEL_Y, VEL_Z,
    ANG_VEL_X, ANG_VEL_Y, ANG_VEL_Z,
    BOOST_AMOUNT, IS_DEMOED, ON_GROUND, HAS_FLIP
) = range(24)

DEFAULT_FEATURES_STR = (
    "cls",
    "is_ball", "is_boost", "is_blue", "is_orange",
    "pos_x", "pos_y", "pos_z",
    "forward_x", "forward_y", "forward_z",
    "up_x", "up_y", "up_z",
    "vel_x", "vel_y", "vel_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "boost_amount", "is_demoed",
    "on_ground", "has_flip"
)

DEFAULT_LABELS = (
    SCORE, NEXT_TOUCH, BOOST_COLLECT, DEMO,
    THROTTLE, STEER, PITCH, YAW, ROLL, JUMP, BOOST, HANDBRAKE,
    RANK
) = range(13)

DEFAULT_LABELS_STR = (
    "score", "next_touch", "boost_collect", "demo",
    "throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake",
    "rank"
)

BALL_COLS = (POS_X, POS_Y, POS_Z, VEL_X, VEL_Y, VEL_Z, ANG_VEL_X, ANG_VEL_Y, ANG_VEL_Z)
BOOST_COLS = (POS_X, POS_Y, POS_Z, BOOST_AMOUNT, IS_DEMOED)
PLAYER_COLS = (POS_X, POS_Y, POS_Z,
               FORWARD_X, FORWARD_Y, FORWARD_Z,
               UP_X, UP_Y, UP_Z,
               VEL_X, VEL_Y, VEL_Z,
               ANG_VEL_X, ANG_VEL_Y, ANG_VEL_Z,
               BOOST_AMOUNT, IS_DEMOED, ON_GROUND, HAS_FLIP)

NORM_TRANSFORM = np.array([
    1.,
    1., 1., 1., 1.,
    2300., 2300., 2300.,
    1., 1., 1.,
    1., 1., 1.,
    2300., 2300., 2300.,
    5.5, 5.5, 5.5,
    100., 1., 1., 1.
])  # x_norm = x / NORM_TRANSFORM

SWAP_TRANSFORM = np.array([
    1.,
    1., 1., 1., 1.,
    -1., -1., 1.,
    -1., -1., 1.,
    -1., -1., 1.,
    -1., -1., 1.,
    -1., -1., 1.,
    1., 1., 1., 1.
])

MIRROR_TRANSFORM = np.array([
    1.,
    1., 1., 1., 1.,
    -1., 1., 1.,
    -1., 1., 1.,
    -1., 1., 1.,
    -1., 1., 1.,
    -1., 1., 1.,
    1., 1., 1., 1.
])
