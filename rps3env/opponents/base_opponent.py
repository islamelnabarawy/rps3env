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
from abc import ABCMeta
from abc import abstractmethod

from rps3env.opponents.match_state import MatchState

__author__ = 'Islam Elnabarawy'


class BaseOpponent:
    __metaclass__ = ABCMeta

    def __init__(self):
        self._state = MatchState()

    @property
    def board(self):
        return self._state.board

    @property
    def captures(self):
        return self._state.captures

    @property
    def counts(self):
        return self._state.counts

    @abstractmethod
    def get_board_layout(self):
        layout = ['R', 'P', 'S'] * 3
        return layout

    def init_board_layout(self, player_side=0, layout=None):
        board = self._state.board

        index = 9 if player_side == 0 else 0
        board['O'][index:index + 9] = ['OU'] * 9

        if layout is None:
            layout = self.get_board_layout()
        index = 0 if player_side == 0 else 9
        board['O'][index:index + 9] = ['P%s' % p for p in layout]

        self._state = MatchState(board)
        return layout

    def reset_board(self, board):
        self._state = MatchState(board)

    @abstractmethod
    def get_player_hand(self):
        hand = 'R'
        return hand

    @abstractmethod
    def get_next_move(self):
        moves = self.get_possible_moves('P')
        return moves[0] if len(moves) > 0 else None

    def apply_move(self, move):
        if 'surrender' in move:
            return
        self._state.apply_move(
            (move['from'][0], int(move['from'][1:])),
            (move['to'][0], int(move['to'][1:])),
            move['outcome'],
            move['otherHand'] if 'otherHand' in move else None
        )

    def get_possible_moves(self, player):
        return ['%s%s:%s%s' % (move[0] + move[1])
                for move in self._state.get_possible_moves(player)]

    def get_move_list(self, ring, index):
        return [('%s%s:%s%s' % (ring, index, move[0], move[1]))
                for move in self._state.get_piece_moves(ring, index)]

    def get_board_hash(self):
        return self._state.get_hash()

    def print_board(self):
        self._state.print_board()
