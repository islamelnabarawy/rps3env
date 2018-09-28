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
import itertools
import logging
import random
import sys
from collections import OrderedDict

import gym
from gym import spaces

import rps3env.config
from rps3env import opponents
from rps3env.classes import PieceType, PlayerColor, Match, BoardPiece

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(rps3env.config.ENV_LOG_LEVEL)
logger.addHandler(logging.StreamHandler(sys.stdout))

BOARD_TEMPLATE = """
                {13}
             {12} {24}  {14}
          {11}        {25} {15}
       {10} {23}              {16}
    {9} {22}         {27}      {26} {17}
       {8}                    {18} {0}
          {7} {21}              {1}
             {6}        {19} {2}
                {5} {20}  {3}
                   {4}
"""

BOARD_POSITIONS = [
    (557, 250), (518, 156), (455, 89), (371, 49), (279, 38), (191, 62), (119, 110), (67, 180), (42, 259),
    (42, 347), (78, 440), (147, 512), (228, 550), (321, 559), (408, 536), (478, 489), (531, 420), (558, 339),
    (463, 234), (380, 141), (255, 128), (158, 191), (122, 305), (172, 421), (283, 475), (400, 444), (467, 355),
    (300, 300)
]


def i2l(i):
    if i < 18:
        return 'O{}'.format(i)
    if i < 27:
        return 'I{}'.format(i - 18)
    return 'C0'


def l2i(l):
    if l[0] == 'O':
        return int(l[1:])
    if l[0] == 'I':
        return 18 + int(l[1:])
    return 27


def action_to_move(action):
    return tuple(i2l(i) for i in action)


def move_to_action(move):
    return tuple(l2i(l) for l in move)


class RPS3GameEnv(gym.Env):
    metadata = {'render.modes': [None, 'human', 'console', 'ansi', 'rgb_array']}

    def __init__(self) -> None:
        super().__init__()
        self._match = None  # type: Match
        self._round = None  # type: int
        self._player_won = None  # type: bool
        self._opponent = None  # type: opponents.BaseOpponent
        self._action_space = None  # type: spaces.MultiDiscrete
        self._observation_space = None  # type: spaces.Tuple
        self._reward_range = None  # type: (int, int)
        self._window = None

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        if self._action_space is None:
            raise ValueError("The environment has not been initialized. Please call reset() first.")
        return self._action_space

    @property
    def observation_space(self) -> spaces.Dict:
        if self._observation_space is None:
            self._observation_space = spaces.Dict([
                ('occupied', spaces.MultiBinary(28)),
                ('player_owned', spaces.MultiBinary(28)),
                ('piece_type', spaces.MultiDiscrete([3] * 28)),
                ('player_captures', spaces.MultiDiscrete([3, 3, 3])),
                ('opponent_captures', spaces.MultiDiscrete([3, 3, 3])),
            ])
        return self._observation_space

    @property
    def reward_range(self) -> (int, int):
        if self._reward_range is None:
            self._reward_range = (-100, 100)
        return self._reward_range

    @property
    def available_actions(self):
        if self._match is None:
            raise ValueError("The environment has not been initialized. Please call reset() first.")
        if self._match.game_over:
            raise ValueError("The current episode is over. Please call reset() to start a new episode.")
        actions = []
        if self._action_space.shape[0] == 9:
            # board setup phase
            actions.extend(list(x) for x in itertools.permutations([1, 2, 3] * 3))
        else:
            # game phase
            color, actions = self._match.get_possible_moves()
        return actions

    def seed(self, seed=None):
        if seed is None:
            # generate the seed the same way random.seed() does internally
            try:
                from os import urandom as _urandom
                seed = int.from_bytes(_urandom(2500), 'big')
            except NotImplementedError:  # pragma: no cover
                import time
                seed = int(time.time() * 256)
        random.seed(seed)
        return seed

    def step(self, action):
        if self._match is None:
            raise ValueError("The environment has not been initialized. Please call reset() first.")
        reward = [0, 0]
        player_move = None
        opponent_move = None
        if self._round < 0:
            assert isinstance(action, list) and len(action) == 9
            self._match.set_board(action, PlayerColor.Blue)
            layout = self._get_opponent_layout()
            self._match.set_board(list(map(lambda v: v.value, layout)), PlayerColor.Red)
            self._action_space = spaces.MultiDiscrete([27, 27])
        else:
            assert isinstance(action, tuple) and len(action) == 2
            player_move = action_to_move(action)
            move_reward, other_piece = self._match.make_move(action[0], action[1], PlayerColor.Blue)
            reward[0] = move_reward

            # check for game over condition
            if self._match.game_over:
                self._player_won = reward[0] > 0
            else:
                # tell opponent about the move's result
                self._opponent_apply_move(action, move_reward, player=True, other_piece=other_piece)

                # make a move for the opponent
                opponent_move = self._get_opponent_move()
                opponent_action = move_to_action(opponent_move)
                move_reward, other_piece = self._match.make_move(opponent_action[0], opponent_action[1],
                                                                 PlayerColor.Red)
                reward[1] = -move_reward

                # check for game over condition
                if self._match.game_over:
                    self._player_won = reward[1] < 0
                else:
                    # tell opponent about the move's result
                    self._opponent_apply_move(opponent_action, move_reward, player=False, other_piece=other_piece)

        self._round += 1
        info = {'round': self._round, 'player_move': player_move, 'opponent_move': opponent_move}
        return self._get_observation(), reward, self._match.game_over, info

    def reset(self):
        self._match = Match()
        self._init_opponent()
        self._round = -1
        self._player_won = False
        self._action_space = spaces.MultiDiscrete([3] * 9)
        return self._get_observation()

    def close(self):
        self.render(close=True)
        super().close()

    def _init_opponent(self):
        self._opponent = opponents.RandomOpponent()

    def render(self, mode='human', close=False):
        if mode not in self.metadata['render.modes']:
            raise gym.error.UnsupportedMode
        if close:
            if self._window is not None:
                self._window.close()
                self._window = None
            return
        if mode == 'ansi':
            return self._get_text_output()
        if mode == 'console':
            print(self._get_text_output())
            return
        if mode == 'human':
            self._render_viewer()
            return
        if mode == 'rgb_array':
            return self._render_viewer(True)

    def _get_text_output(self):
        output = BOARD_TEMPLATE.format(*[
            ('{}!' if p.revealed else '{}').format(p.to_str(PlayerColor.Blue, False))
            if p is not None else '..' for p in self._match.board
        ])
        output += self._opponent.print_board(output=False)
        return output

    def _get_observation(self):
        obs = OrderedDict([
            ('occupied', [p is not None for p in self._match.board]),
            ('player_owned', [p is not None and p.color == PlayerColor.Blue for p in self._match.board]),
            ('piece_type', [
                PieceType.N.value if p is None else
                (p.piece_type.value if p.color == PlayerColor.Blue or p.revealed else PieceType.U.value)
                for p in self._match.board
            ]),
            ('player_captures', [0, 0, 0]),
            ('opponent_captures', [0, 0, 0]),
        ])
        if self._round < 0:
            return obs
        player_counts = [0, 0, 0]
        opponent_counts = [0, 0, 0]
        for p in [p for p in self._match.board if p is not None]:
            if p.color == PlayerColor.Blue:
                player_counts[p.piece_type.value - 1] += 1
            else:
                opponent_counts[p.piece_type.value - 1] += 1
        obs['player_captures'] = [3 - x for x in player_counts]
        obs['opponent_captures'] = [3 - x for x in opponent_counts]
        return obs

    def _get_opponent_layout(self):
        layout = self._opponent.init_board_layout(1)
        return [PieceType[s] for s in layout]

    def _get_opponent_move(self):
        opponent_move = self._opponent.get_next_move().split(':')
        logger.debug("opponent move: %s", opponent_move)
        return opponent_move

    def _opponent_apply_move(self, move, result, player, other_piece):
        move_from = move[0]
        move_to = move[1]
        from_piece = self._match.board[move_from]
        to_piece = self._match.board[move_to]
        move_data = {'from': i2l(move_from), 'to': i2l(move_to)}
        if result == 0:
            if from_piece is None:
                move_data['outcome'] = 'M'  # this was a move action
            else:
                move_data['outcome'] = 'T'  # it was a tie
                move_data['otherHand'] = other_piece.name
        else:
            if result > 0:
                move_data['outcome'] = 'W'
                if player:
                    move_data['otherHand'] = to_piece.piece_type.name
                else:
                    move_data['otherHand'] = PieceType(((to_piece.piece_type.value + 1) % 3) + 1).name
            else:
                move_data['outcome'] = 'L'
                if player:
                    move_data['otherHand'] = PieceType(((to_piece.piece_type.value + 1) % 3) + 1).name
                else:
                    move_data['otherHand'] = to_piece.piece_type.name

        self._opponent.apply_move(move_data)

    def _render_viewer(self, return_rgb_array=False):
        import numpy as np
        import pyglet
        from pyglet import gl

        if self._window is None:
            import os
            self._bg = pyglet.image.load(os.path.join(os.path.dirname(__file__), '../assets/board.png'))
            self._window = pyglet.window.Window(
                width=rps3env.config.VIEWER_WIDTH,
                height=rps3env.config.VIEWER_HEIGHT
            )  # type: pyglet.window

            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glClearColor(1, 1, 1, 1)
        self._window.clear()
        self._window.switch_to()
        self._window.dispatch_events()
        board_offset_x = 50
        board_offset_y = 50

        def draw_piece(index: int, piece: BoardPiece):
            x, y = BOARD_POSITIONS[index]
            color = (0, 0, 255, 255) if piece.color == PlayerColor.Blue else (255, 0, 0, 255)
            pyglet.text.Label(
                ('{}!' if piece.revealed else '{}').format(piece.piece_type.name),
                font_name='Arial', font_size=28, anchor_x='center', anchor_y='center',
                x=x + board_offset_x, y=y + board_offset_y, color=color
            ).draw()

        self._bg.blit(board_offset_x, board_offset_y, width=600, height=600)

        for i, p in enumerate(self._match.board):
            if p is not None:
                draw_piece(i, p)

        if self._match.game_over:
            txt = "Game Over! {} won.".format('Player' if self._player_won else 'Opponent')
            label = pyglet.text.Label(
                txt, font_name='Arial', font_size=32, anchor_x='center', anchor_y='center',
                x=350, y=30, color=(0, 0, 0, 255)
            )
            label.draw()

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            arr = np.fromstring(buffer.get_image_data().data, dtype=np.uint8, sep='')
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self._window.flip()

        return arr


class RPS3GameMinMaxEnv(RPS3GameEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._opponent_kwargs = kwargs

    @property
    def settings(self):
        return self._opponent_kwargs

    def _init_opponent(self):
        self._opponent = opponents.MinMaxOpponent(**self._opponent_kwargs)
