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
import copy
import logging
import math
import sys

import gym
import numpy as np

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

STARTING_BOARD = {
    'O': [
        '0', '0', '0', '0', '0', '0', '0', '0', '0',
        '0', '0', '0', '0', '0', '0', '0', '0', '0'
    ],
    'I': ['0', '0', '0', '0', '0', '0', '0', '0', '0'],
    'C': ['0']
}

BOARD_TEMPLATE = """
            O13
         O12 I6  O14
      O11        I7 O15
   O10 I5              O16
O9 I4         C0      I8 O17
   O8                    I0 O0
      O7 I3              O1
         O6        I1 O2
            O5 I2  O3
               O4
"""


class RPS3GameEnv(gym.Env):
    metadata = {'render.modes': [None, 'human', 'ansi']}

    def __init__(self) -> None:
        super().__init__()
        self.board = None
        self.turn = None

    def _step(self, action):
        if self.turn < 0:
            assert(isinstance(action, list))
            for i, v in enumerate(action):
                self.board['O'][i] = 'P'+v
            for i in range(9, 18):
                self.board['O'][i] = 'OU'
        else:
            assert(isinstance(action, tuple) and len(action) == 2)
            assert(action in self.get_player_moves())

        self.turn += 1
        return self._get_observation(), 0, False, {'turn': self.turn}

    def _reset(self):
        self.board = copy.deepcopy(STARTING_BOARD)
        self.turn = -1
        return self._get_observation()

    def _render(self, mode='human', close=False):
        if close:
            return
        output = BOARD_TEMPLATE
        for ring in self.board.keys():
            for index in range(len(self.board[ring]) - 1, -1, -1):
                value = self.board[ring][index]
                output = output.replace("%s%d" % (ring, index), '..' if value == '0' else value)

        if mode == 'human':
            print(output)
        elif mode == 'ansi':
            return output

    def _get_observation(self):
        return self.board['O'] + self.board['I'] + self.board['C']

    def get_player_moves(self, player='P'):
        moves = []
        for (ring, squares) in self.board.items():
            for (index, piece) in enumerate(squares):
                if piece[0] == player:
                    moves.extend([
                        ('{}{}'.format(ring, index), '{}{}'.format(r, i))
                        for (r, i) in self.get_piece_moves(ring, index)
                    ])
        return moves

    def get_piece_moves(self, ring, index):
        result = []
        player = self.board[ring][index][0]
        if player == '0':
            return result

        if ring == 'O':
            left = (index + 1) % 18
            if self.board[ring][left][0] != player:
                result.append((ring, left))
            right = (index + 17) % 18
            if self.board[ring][right][0] != player:
                result.append((ring, right))
            inside = int(math.floor(index / 2.0))
            if self.board['I'][inside][0] != player:
                result.append(('I', inside))
        elif ring == 'I':
            left = (index + 1) % 9
            if self.board[ring][left][0] != player:
                result.append((ring, left))
            right = (index + 8) % 9
            if self.board[ring][right][0] != player:
                result.append((ring, right))
            outside = int(np.math.floor(index * 2))
            if self.board['O'][outside][0] != player:
                result.append(('O', outside))
            outside += 1
            if self.board['O'][outside][0] != player:
                result.append(('O', outside))
            if self.board['C'][0][0] != player:
                result.append(('C', 0))
        elif ring == 'C':
            result.extend(
                [('I', i) for (i, v) in
                 enumerate(self.board['I']) if v[0] != player]
            )

        return result
