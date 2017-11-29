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

import gym
import numpy as np

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    def _step(self, action):
        pass

    def _reset(self):
        return np.zeros([28, 1], dtype=np.int8)

    def _render(self, mode='human', close=False):
        board = STARTING_BOARD
        output = BOARD_TEMPLATE
        for ring in board.keys():
            for index in range(len(board[ring]) - 1, -1, -1):
                value = board[ring][index]
                output = output.replace("%s%d" % (ring, index), '..' if value == '0' else value)

        if mode == 'human':
            print(output)
        elif mode == 'ansi':
            return output
