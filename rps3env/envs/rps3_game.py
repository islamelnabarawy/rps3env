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
import math
import random
import sys

import gym

from rps3env import opponents
from rps3env.classes import PieceType, BoardPiece, BoardLocation

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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
        self.board = None  # type: dict[str, list[BoardLocation]]
        self.turn = None  # type: int
        self.opponent = None  # type: opponents.BaseOpponent

    def _seed(self, seed=None):
        if seed is None:
            # generate the seed the same way random.seed() does internally
            try:
                from os import urandom as _urandom
                seed = int.from_bytes(_urandom(2500), 'big')
            except NotImplementedError:
                import time
                seed = int(time.time() * 256)
        random.seed(seed)
        return seed

    def _step(self, action):
        reward = [0, 0]
        done = False
        if self.turn < 0:
            assert isinstance(action, list) and len(action) == 9
            assert action.count('R') == action.count('P') == action.count('S') == 3
            for i, v in enumerate(action):
                self.board['O'][i].piece = BoardPiece(PieceType[v], True)
            layout = self._get_opponent_layout()
            for i in range(9, 18):
                self.board['O'][i].piece = BoardPiece(layout[i - 9], False)
        else:
            assert isinstance(action, tuple) and len(action) == 2
            assert action in self._get_player_moves()
            reward[0] = self._make_move(action)

            # tell opponent about the move's result
            self._opponent_apply_move(action, reward[0])

            # check game over condition
            done, player_won = self._is_game_over()
            if done:
                reward[0] = 100 if player_won else -100
            else:
                # make a move for the opponent
                move = self._get_opponent_move()
                reward[1] = -self._make_move(move)

                # tell opponent about the move's result
                self._opponent_apply_move(move, -reward[1])

                # check game over condition
                done, player_won = self._is_game_over()
                if done:
                    reward[1] = 100 if player_won else -100

        self.turn += 1
        return self._get_observation(), reward, done, {'turn': self.turn}

    def _reset(self):
        self._init_board()
        self._init_opponent()
        self.turn = -1
        return self._get_observation()

    def _init_board(self):
        self.board = {
            'O': [BoardLocation('O', i) for i in range(18)],
            'I': [BoardLocation('I', i) for i in range(9)],
            'C': [BoardLocation('C', i) for i in range(1)],
        }

    def _init_opponent(self):
        self.opponent = opponents.RandomOpponent()

    def _render(self, mode='human', close=False):
        if close:
            return
        output = BOARD_TEMPLATE
        for ring in 'OIC':
            for index in range(len(self.board[ring]) - 1, -1, -1):
                location = self.board[ring][index]
                output = output.replace(
                    "%s%d" % (ring, index),
                    '..' if location.piece is None else ('{}!' if location.piece.revealed else '{}').format(
                        location.piece.to_str(False))
                )

        if mode == 'human':
            print(output)
        elif mode == 'ansi':
            return output

    def _get_observation(self):
        board = self.board['O'] + self.board['I'] + self.board['C']
        obs = []
        for i, l in enumerate(board):
            obs.append(str(l.piece) if l.piece is not None else '0')
        return obs

    def _get_opponent_layout(self):
        layout = self.opponent.init_board_layout(1)
        return [PieceType[s] for s in layout]

    def _get_opponent_move(self):
        opponent_move = self.opponent.get_next_move().split(':')
        logger.debug("opponent move: %s", opponent_move)
        return opponent_move

    def _opponent_apply_move(self, move, result):
        move_from = move[0]
        move_to = move[1]
        from_location = self.board[move_from[0]][int(move_from[1:])]
        to_location = self.board[move_to[0]][int(move_to[1:])]
        move_data = {'from': move_from, 'to': move_to}
        if result == 0:
            if from_location.piece is None:
                move_data['outcome'] = 'M'  # this was a move action
            else:
                move_data['outcome'] = 'T'  # it was a tie
                move_data['otherHand'] = from_location.piece.piece_type.name
        else:
            move_data['outcome'] = 'W' if result > 0 else 'L'
            move_data['otherHand'] = from_location.piece.piece_type.name if from_location.piece is not None \
                else to_location.piece.piece_type.name
        self.opponent.apply_move(move_data)

    def _get_player_moves(self, player=True):
        moves = []
        for ring in 'OIC':
            squares = self.board[ring]
            for (index, location) in enumerate(squares):
                if location.piece is not None and location.piece.player_owned == player:
                    moves.extend([
                        ('{}{}'.format(ring, index), '{}{}'.format(r, i))
                        for (r, i) in self._get_piece_moves(ring, index)
                    ])
        return moves

    def _get_piece_moves(self, ring, index):
        result = []
        piece = self.board[ring][index].piece
        if piece is None:
            return result

        def empty_or_opponent(location):
            return location.piece is None or location.piece.player_owned != piece.player_owned

        if ring == 'O':
            left = (index + 1) % 18
            if empty_or_opponent(self.board[ring][left]):
                result.append((ring, left))
            right = (index + 17) % 18
            if empty_or_opponent(self.board[ring][right]):
                result.append((ring, right))
            inside = int(math.floor(index / 2.0))
            if empty_or_opponent(self.board['I'][inside]):
                result.append(('I', inside))
        elif ring == 'I':
            left = (index + 1) % 9
            if empty_or_opponent(self.board[ring][left]):
                result.append((ring, left))
            right = (index + 8) % 9
            if empty_or_opponent(self.board[ring][right]):
                result.append((ring, right))
            outside = int(math.floor(index * 2))
            if empty_or_opponent(self.board['O'][outside]):
                result.append(('O', outside))
            outside += 1
            if empty_or_opponent(self.board['O'][outside]):
                result.append(('O', outside))
            if empty_or_opponent(self.board['C'][0]):
                result.append(('C', 0))
        elif ring == 'C':
            result.extend(
                [('I', i) for (i, v) in enumerate(self.board['I']) if empty_or_opponent(v)]
            )

        return result

    def _make_move(self, action):
        move_from, move_to = action
        from_ring = move_from[0]
        from_index = int(move_from[1:])
        to_ring = move_to[0]
        to_index = int(move_to[1:])
        from_location = self.board[from_ring][from_index]
        to_location = self.board[to_ring][to_index]
        if to_location.piece is None:
            # swap the piece between locations
            from_location.piece, to_location.piece = None, from_location.piece
            return 0

        attacking_hand = from_location.piece.piece_type
        defending_hand = to_location.piece.piece_type
        result = 0

        # no matter what the outcome is, both pieces will be revealed
        from_location.piece.revealed = True
        to_location.piece.revealed = True

        if attacking_hand == defending_hand:
            return result

        if attacking_hand == PieceType.R:
            result = 1 if defending_hand == PieceType.S else -1
        if attacking_hand == PieceType.P:
            result = 1 if defending_hand == PieceType.R else -1
        if attacking_hand == PieceType.S:
            result = 1 if defending_hand == PieceType.P else -1

        if result > 0:
            # swap the piece between locations
            from_location.piece, to_location.piece = None, from_location.piece
        elif result < 0:
            from_location.piece = None

        return result

    def _is_game_over(self):
        piece_counters = {PieceType.R: PieceType.P, PieceType.P: PieceType.S, PieceType.S: PieceType.R}
        player_counts = {PieceType.R: 0, PieceType.P: 0, PieceType.S: 0}
        opponent_counts = {PieceType.R: 0, PieceType.P: 0, PieceType.S: 0}
        for ring in 'OIC':
            for piece in [location.piece for location in self.board[ring] if location.piece is not None]:
                if piece.player_owned:
                    player_counts[piece.piece_type] += 1
                else:
                    opponent_counts[piece.piece_type] += 1
        if sum(player_counts.values()) == 0:
            return True, False
        if sum(opponent_counts.values()) == 0:
            return True, True
        center_piece = self.board['C'][0].piece
        if center_piece is not None:
            center_counter = piece_counters[center_piece.piece_type]
            if center_piece.player_owned and opponent_counts[center_counter] == 0:
                return True, True
            if not center_piece.player_owned and player_counts[center_counter] == 0:
                return True, False

        return False, False


class RPS3GameMinMaxEnv(RPS3GameEnv):
    def _init_opponent(self):
        self.opponent = opponents.MinMaxOpponent()
