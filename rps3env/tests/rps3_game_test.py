"""
   Copyright 2017 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import logging
import random
import sys
import unittest

import gym
from gym import Space, spaces

import rps3env.config
from rps3env.envs import RPS3GameEnv, RPS3GameMinMaxEnv
from rps3env.tests.utils import captured_output

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(rps3env.config.ENV_LOG_LEVEL)
logger.addHandler(logging.StreamHandler(sys.stdout))

EMPTY_BOARD = """
                ..
             .. ..  ..
          ..        .. ..
       .. ..              ..
    .. ..         ..      .. ..
       ..                    .. ..
          .. ..              ..
             ..        .. ..
                .. ..  ..
                   ..

                    ..
                 .. ..  ..
              ..        .. ..
           .. ..              ..
        .. ..         ..      .. ..
           ..                    .. ..
              .. ..              ..
                 ..        .. ..
                    .. ..  ..
                       ..
        
Turns: 0
Player Counts: [0, 0, 0]
Player Reveals: [0, 0, 0]
Opponent Captures: [0, 0, 0]
Opponent Counts: [0, 0, 0, 0]
Probabilities: [0.0, 0.0, 0.0]"""

INIT_BOARD = """
                OP
             OR ..  OS
          OP        .. OR
       OS ..              OS
    OP ..         ..      .. OR
       PS                    .. PR
          PP ..              PP
             PR        .. PS
                PS ..  PR
                   PP

                    PP
                 PR ..  PS
              PP        .. PR
           PS ..              PS
        PP ..         ..      .. PR
           OU                    .. OU
              OU ..              OU
                 OU        .. OU
                    OU ..  OU
                       OU
        
Turns: 0
Player Counts: [3, 3, 3]
Player Reveals: [0, 0, 0]
Opponent Captures: [0, 0, 0]
Opponent Counts: [0, 0, 0, 9]
Probabilities: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]"""

OBS_BEFORE_BOARD_INIT = {
    'occupied': [False] * 28,
    'player_owned': [False] * 28,
    'piece_type': [-1] * 28
}

OBS_AFTER_BOARD_INIT = {
    'occupied': [True] * 18 + [False] * 10,
    'player_owned': [True] * 9 + [False] * 19,
    'piece_type': [1, 2, 3] * 3 + [0] * 9 + [-1] * 10
}

OBS_AFTER_LEGAL_MOVE = {
    'occupied': [False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True,
                 True, True, False, False, False, False, False, False, True, False, False],
    'player_owned': [False, True, True, True, True, True, True, True, True, False, False, False, False, False, False,
                     False, False, False, True, False, False, False, False, False, False, False, False, False],
    'piece_type': [-1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, -1, -1, -1, -1, -1, -1, 0, -1, -1]
}

OBS_AFTER_CHALLENGE_WIN = {
    'occupied': [True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False,
                 True, False, False, False, False, False, False, False, False, True, False],
    'player_owned': [True, True, True, True, True, True, True, True, False, True, False, False, False, False, False,
                     False, False, False, False, False, False, False, False, False, False, False, False, False],
    'piece_type': [1, 2, 3, 1, 2, 3, 1, 2, -1, 3, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1]
}

OBS_AFTER_CHALLENGE_TIE = {
    'occupied': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True,
                 True, False, False, False, False, False, False, False, True, False, False],
    'player_owned': [True, True, True, True, True, True, True, True, True, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False, False, False, False, False, False],
    'piece_type': [1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0, 0, 0, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1]
}

OBS_AFTER_CHALLENGE_LOSS = {
    'occupied': [True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True,
                 True, False, False, False, False, False, False, False, False, False, False],
    'player_owned': [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False, False, False, False, False, False],
    'piece_type': [1, 2, 3, 1, 2, 3, 1, 2, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
}

OBS_AFTER_FULL_GAME = {
    'occupied': [True, True, True, True, True, True, False, True, False, False, False, False, False, True, False, False,
                 True, False, True, False, False, False, False, True, True, True, True, True],
    'player_owned': [True, True, True, True, True, True, False, True, False, False, False, False, False, True, False,
                     False, False, False, False, False, False, False, False, False, False, False, False, True],
    'piece_type': [1, 2, 3, 1, 2, 3, -1, 2, -1, -1, -1, -1, -1, 3, -1, -1, 0, -1, 0, -1, -1, -1, -1, 3, 0, 0, 1, 1]
}

OBS_AFTER_TAKE_ALL_PIECES = {
    'occupied': [False, False, True, False, True, True, True, True, False, False, False, False, False, False, False,
                 False, False, True, False, False, False, False, False, False, True, False, False, False],
    'piece_type': [-1, -1, 3, -1, 2, 3, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 2, -1, -1,
                   -1],
    'player_owned': [False, False, True, False, True, True, True, True, False, False, False, False, False, False, False,
                     False, False, True, False, False, False, False, False, False, True, False, False, False]
}

OBS_AFTER_LOSE_ALL_PIECES = {
    'occupied': [True, False, False, False, False, False, False, True, False, False, False, True, False, True, False,
                 False, True, True, False, False, False, False, False, True, False, False, False, False],
    'piece_type': [1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, 0, -1, 0, -1, -1, 0, 1, -1, -1, -1, -1, -1, 2, -1, -1, -1,
                   -1],
    'player_owned': [False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False, False, False, False, False, False, False]
}

AVAILABLE_ACTIONS_AFTER_INIT = [
    (0, 17), (0, 18), (1, 18), (2, 19), (3, 19), (4, 20), (5, 20), (6, 21), (7, 21), (8, 9), (8, 22)
]


class RPS3GameEnvTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('RPS3Game-v0')  # type: RPS3GameEnv

    def tearDown(self):
        self.env.close()

    def init_board(self):
        self.env.reset()
        obs, reward, done, info = self.env.step([1, 2, 3] * 3)
        return obs, reward, done, info

    def step_assert(self, obs_actual, reward_actual, done_actual, info_actual,
                    obs_expected, reward_expected=None, done_expected=False, info_expected=None):
        if reward_expected is None:
            reward_expected = [0, 0]
        self.assertEqual(obs_expected, obs_actual)
        self.assertEqual(reward_expected, reward_actual)
        self.assertEqual(done_expected, done_actual)
        if info_expected is not None:
            for k in info_expected.keys():
                self.assertEqual(info_expected[k], info_actual[k])

    def test_initializable(self):
        self.assertIsNotNone(self.env, msg='gym.make() returned None.')

    def test_action_space_pre_reset(self):
        self.assertRaises(ValueError, lambda: self.env.action_space)

    def test_action_space_pre_init(self):
        self.env.reset()
        self.assertIsInstance(self.env.action_space, Space)
        self.assertEqual(9, self.env.action_space.shape)
        self.assertEqual(3, max(self.env.action_space.high))
        self.assertEqual(3, min(self.env.action_space.high))
        self.assertEqual(1, max(self.env.action_space.low))
        self.assertEqual(1, min(self.env.action_space.low))

    def test_action_space_post_init(self):
        self.init_board()
        self.assertIsInstance(self.env.action_space, Space)
        self.assertEqual(2, self.env.action_space.shape)
        self.assertEqual(27, max(self.env.action_space.high))
        self.assertEqual(27, min(self.env.action_space.high))
        self.assertEqual(0, max(self.env.action_space.low))
        self.assertEqual(0, min(self.env.action_space.low))

    def test_observation_space(self):
        self.assertIsInstance(self.env.observation_space, spaces.Dict)
        self.assertEqual(3, len(self.env.observation_space.spaces))
        self.assertEqual(28, self.env.observation_space.spaces['occupied'].n)
        self.assertEqual(28, self.env.observation_space.spaces['player_owned'].n)
        self.assertEqual(28, self.env.observation_space.spaces['piece_type'].shape)
        self.assertEqual(3, max(self.env.observation_space.spaces['piece_type'].high))
        self.assertEqual(3, min(self.env.observation_space.spaces['piece_type'].high))
        self.assertEqual(-1, max(self.env.observation_space.spaces['piece_type'].low))
        self.assertEqual(-1, min(self.env.observation_space.spaces['piece_type'].low))

    def test_reward_range(self):
        self.assertEqual(-100, self.env.reward_range[0])
        self.assertEqual(100, self.env.reward_range[1])

    def test_step_before_reset(self):
        self.assertRaises(ValueError, lambda: self.env.step([1, 2, 3] * 3))

    def test_reset(self):
        actual = self.env.reset()
        expected = OBS_BEFORE_BOARD_INIT
        self.assertEqual(expected, actual)

    def test_render_empty_board(self):
        self.env.reset()
        actual = self.env.render(mode='ansi')
        self.assertEqual(EMPTY_BOARD, actual)

    def test_output_empty_board(self):
        self.env.reset()
        with captured_output() as (out, err):
            self.env.render(mode='console')
        self.assertEqual(EMPTY_BOARD, out.getvalue().rstrip())

    def test_set_board(self):
        self.env.seed()
        obs, reward, done, info = self.init_board()
        self.step_assert(obs, reward, done, info, OBS_AFTER_BOARD_INIT)

    def test_render_set_board(self):
        self.env.seed(0)
        self.init_board()
        actual = self.env.render(mode='ansi')
        self.assertEqual(INIT_BOARD, actual)

    def test_render_human(self):
        self.env.seed(0)
        self.init_board()
        self.env.render(mode='human')
        done = False
        while not done:
            self.env.render(mode='human')
            action = random.choice(self.env.available_actions)
            obs, reward, done, info = self.env.step(action)
        self.env.render(mode='human')

    def test_render_rgb_array(self):
        self.env.seed(0)
        self.init_board()
        self.env.render(mode='rgb_array')
        done = False
        while not done:
            self.env.render(mode='rgb_array')
            action = random.choice(self.env.available_actions)
            obs, reward, done, info = self.env.step(action)
        self.env.render(mode='rgb_array')

    def test_render_invalid_mode(self):
        self.init_board()
        self.assertRaises(gym.error.UnsupportedMode, lambda: self.env.render('invalid'))

    def test_pre_init_available_actions(self):
        self.assertRaises(ValueError, lambda: self.env.available_actions)

    def test_init_available_actions(self):
        self.env.reset()
        available_actions = self.env.available_actions
        random.seed(0)
        self.assertIsInstance(random.choice(available_actions), list)
        self.assertEqual(362880, len(available_actions))

    def test_game_available_actions(self):
        self.init_board()
        available_actions = self.env.available_actions
        self.assertEqual(AVAILABLE_ACTIONS_AFTER_INIT, available_actions)

    def test_post_game_available_actions(self):
        self.init_board()
        done = False
        while not done:
            _, _, done, _ = self.env.step(random.choice(self.env.available_actions))
        self.assertRaises(ValueError, lambda: self.env.available_actions)

    def test_malformed_move(self):
        self.env.seed(0)
        self.init_board()

        self.assertRaises(AssertionError, lambda: self.env.step(()))
        self.assertRaises(AssertionError, lambda: self.env.step((0,)))

    def test_illegal_moves(self):
        self.env.seed(0)
        self.init_board()

        self.assertRaises(AssertionError, lambda: self.env.step((0, 1)))
        self.assertRaises(AssertionError, lambda: self.env.step((0, 19)))
        self.assertRaises(AssertionError, lambda: self.env.step((0, 27)))

    def test_legal_move(self):
        self.env.seed(0)
        self.init_board()
        obs, reward, done, info = self.env.step((0, 18))

        self.step_assert(obs, reward, done, info, OBS_AFTER_LEGAL_MOVE, info_expected={'round': 1})

    def test_challenge_tie(self):
        self.env.seed(0)
        self.init_board()
        obs, reward, done, info = self.env.step((0, 17))

        self.step_assert(obs, reward, done, info, OBS_AFTER_CHALLENGE_TIE, info_expected={'round': 1})

    def test_challenge_win(self):
        self.env.seed(0)
        self.init_board()
        obs, reward, done, info = self.env.step((8, 9))

        self.step_assert(obs, reward, done, info, OBS_AFTER_CHALLENGE_WIN,
                         reward_expected=[1, 0], info_expected={'round': 1})

    def test_challenge_loss(self):
        self.env.seed(2)
        self.init_board()
        obs, reward, done, info = self.env.step((8, 9))

        self.step_assert(obs, reward, done, info, OBS_AFTER_CHALLENGE_LOSS,
                         reward_expected=[-1, 0], info_expected={'round': 1})

    def test_full_game(self):
        self.env.seed(0)
        self.init_board()
        moves = [(8, 9), (9, 22), (22, 23), (23, 11), (11, 23), (23, 11), (11, 12), (12, 13), (6, 21)]
        for move in moves:
            self.env.step(move)
        obs, reward, done, info = self.env.step((21, 27))

        self.step_assert(obs, reward, done, info, OBS_AFTER_FULL_GAME,
                         reward_expected=[100, 0], done_expected=True, info_expected={'round': 10})

    def test_full_game_take_all_pieces(self):
        self.env.seed(0)
        self.env.reset()
        self.env.step([2, 1, 3, 1, 2, 3, 1, 2, 3])
        moves = [(0, 17), (1, 18), (18, 26), (26, 16), (16, 26), (26, 25), (17, 16), (16, 15), (25, 14), (8, 22),
                 (22, 23), (23, 11), (11, 23), (23, 24), (24, 13), (13, 24), (3, 19), (19, 18), (18, 26), (26, 17),
                 (15, 25)]
        for move in moves:
            self.env.step(move)
        obs, reward, done, info = self.env.step((25, 24))
        self.step_assert(obs, reward, done, info, OBS_AFTER_TAKE_ALL_PIECES,
                         reward_expected=[100, 0], done_expected=True, info_expected={'round': len(moves) + 1})

    def test_full_game_lose_all_pieces(self):
        self.env.seed(0)
        self.env.reset()
        self.env.step([3, 2, 1, 3, 2, 1, 3, 2, 1])
        moves = [(0, 17), (8, 9), (3, 19), (19, 27), (27, 25), (7, 21), (21, 27), (1, 18), (18, 27), (4, 20), (20, 21),
                 (2, 19), (19, 27), (6, 7), (5, 6), (7, 8), (8, 22), (6, 7), (7, 8), (8, 9), (22, 27), (9, 22),
                 (22, 23), (27, 26)]
        for move in moves:
            self.env.step(move)
        obs, reward, done, info = self.env.step((26, 17))
        self.step_assert(obs, reward, done, info, OBS_AFTER_LOSE_ALL_PIECES,
                         reward_expected=[-100, 0], done_expected=True, info_expected={'round': len(moves) + 1})


class RPS3GameMinMaxEnvTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('RPS3Game-v1')  # type: RPS3GameMinMaxEnv

    def tearDown(self):
        self.env.close()

    def test_random_play_level_1_1(self):
        self.play_randomly(0, 1, 27)

    def test_random_play_level_1_2(self):
        self.play_randomly(3, 1, 38)

    def test_random_play_level_2_1(self):
        self.play_randomly(0, 2, 40)

    def test_random_play_level_2_2(self):
        self.play_randomly(2, 2, 45)

    def test_random_play_level_3_1(self):
        self.play_randomly(0, 3, 61)

    def test_random_play_level_3_2(self):
        self.play_randomly(3, 3, 43)

    def test_history_table_printing(self):
        self.play_randomly(0, 1, 27)
        with captured_output() as (out, err):
            self.env._opponent.print_history_table()
        as_str = self.env._opponent.get_history_table()
        as_out = out.getvalue().rstrip()
        self.assertEqual(as_out, as_str)

    def play_randomly(self, seed, depth_limit, final_round):
        self.env.settings['depth_limit'] = depth_limit
        self.env.reset()
        self.env.seed(seed)
        done = False
        reward, info = None, None
        total_reward = 0
        while not done:
            logger.debug(self.env.render(mode='ansi'))
            action = random.choice(self.env.available_actions)
            logger.debug('Taking action: %s', action)
            obs, reward, done, info = self.env.step(action)
            logger.debug('Got reward: %s', reward)
            logger.debug('Info: %s', info)
            total_reward += sum(reward)
        logger.debug(self.env.render(mode='ansi'))
        logger.debug('Game over. Total reward: %d', total_reward)
        self.assertEqual([0, -100], reward)
        self.assertTrue(done)
        self.assertEqual(final_round, info['round'])
