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
import math

__author__ = 'Islam Elnabarawy'


class MatchState:
    PIECE_KEY = ['R', 'P', 'S', 'U']

    STARTING_BOARD = {
        'O': [
            '0', '0', '0', '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0', '0', '0', '0'
        ],
        'I': ['0', '0', '0', '0', '0', '0', '0', '0', '0'],
        'C': ['0']
    }

    def __init__(self, board=None, captures=None, opponent_counts=None,
                 player_counts=None, player_reveals=None, turns=0):
        self._board = copy.deepcopy(board) \
            if board is not None else copy.deepcopy(MatchState.STARTING_BOARD)
        self._captures = captures \
            if captures is not None else [0] * 3
        self._opponent_counts = opponent_counts \
            if opponent_counts is not None else [0] * 3 + [9]
        self._player_counts = player_counts \
            if player_counts is not None else [0] * 3
        self._player_reveals = player_reveals \
            if player_reveals is not None else [0] * 3
        self._turns = turns
        if opponent_counts is None or player_counts is None or player_reveals is None:
            self._update_counts()

    def clone(self):
        """
        :rtype : MatchState
        :return: A cloned copy of this object
        """
        return copy.deepcopy(self)

    def get_board_value(self, ring, index):
        return self._board[ring][index]

    @property
    def board(self):
        return copy.deepcopy(self._board)

    @property
    def captures(self):
        return copy.copy(self._captures)

    @property
    def counts(self):
        return copy.copy(self._opponent_counts)

    @property
    def player_counts(self):
        return copy.copy(self._player_counts)

    @property
    def player_reveals(self):
        return copy.copy(self._player_reveals)

    @property
    def turns(self):
        return self._turns

    def _update_counts(self):
        self._opponent_counts = self._captures + [0]
        self._player_counts = [0] * 3
        for (ring, squares) in self._board.items():
            for (index, piece) in enumerate(squares):
                if piece[0] == 'O':
                    self._opponent_counts[self.PIECE_KEY.index(piece[1])] += 1
                elif piece[0] == 'P':
                    self._player_counts[self.PIECE_KEY.index(piece[1])] += 1
                    if len(piece) > 2:
                        self._player_reveals[self.PIECE_KEY.index(piece[1])] += 1

    def apply_move(self, move_from, move_to, outcome, other_hand=None):
        self._turns += 1
        from_ring, from_index = move_from
        to_ring, to_index = move_to
        from_piece = self._board[from_ring][from_index]
        to_piece = self._board[to_ring][to_index]
        other_index = MatchState.PIECE_KEY.index(other_hand) \
            if other_hand is not None else None

        if outcome == 'M':
            self._board[to_ring][to_index] = from_piece
            self._board[from_ring][from_index] = '0'
        elif outcome == 'W':
            self._board[to_ring][to_index] = from_piece
            self._board[from_ring][from_index] = '0'
            if from_piece[0] == 'O':
                self._player_counts[self.PIECE_KEY.index(to_piece[1])] -= 1
                if from_piece[1] == 'U':
                    # this piece is now known, update the counts
                    self._board[to_ring][to_index] = 'O' + other_hand
                    self._opponent_counts[other_index] += 1
                    self._opponent_counts[3] -= 1
                if len(to_piece) < 3:
                    # my piece was unknown, update the reveals count
                    self._player_reveals[self.PIECE_KEY.index(to_piece[1])] += 1
            else:
                self._captures[other_index] += 1
                if to_piece[1] == 'U':
                    # this piece is now known, update the counts
                    self._opponent_counts[other_index] += 1
                    self._opponent_counts[3] -= 1
                if len(from_piece) < 3:
                    # my piece was unknown, update the reveals count
                    self._player_reveals[self.PIECE_KEY.index(from_piece[1])] += 1
                    # mark the piece on the board as revealed
                    self._board[to_ring][to_index] = from_piece + '!'
        elif outcome == 'T':
            ring, index = (to_ring, to_index) if from_piece[0] == 'P' else (from_ring, from_index)
            opponent_piece = self._board[ring][index]
            if opponent_piece[1] == 'U':
                # this piece is now known, update the counts
                self._board[ring][index] = 'O' + other_hand
                self._opponent_counts[other_index] += 1
                self._opponent_counts[3] -= 1
            ring, index = (to_ring, to_index) if from_piece[0] == 'O' else (from_ring, from_index)
            player_piece = self._board[ring][index]
            if len(player_piece) < 3:
                # my piece was unknown, update the reveals count
                self._player_reveals[self.PIECE_KEY.index(player_piece[1])] += 1
                # mark the piece on the board as revealed
                self._board[ring][index] = player_piece + '!'
        elif outcome == 'L':
            self._board[from_ring][from_index] = '0'
            if from_piece[0] == 'P':
                self._player_counts[self.PIECE_KEY.index(from_piece[1])] -= 1
                if to_piece[1] == 'U':
                    # this piece is now known, update the counts
                    self._board[to_ring][to_index] = 'O' + other_hand
                    self._opponent_counts[other_index] += 1
                    self._opponent_counts[3] -= 1
                if len(from_piece) < 3:
                    # my piece was unknown, update the reveals count
                    self._player_reveals[self.PIECE_KEY.index(from_piece[1])] += 1
            else:
                self._captures[other_index] += 1
                if from_piece[1] == 'U':
                    # this piece is now known, update the counts
                    self._opponent_counts[other_index] += 1
                    self._opponent_counts[3] -= 1
                if len(to_piece) < 3:
                    # my piece was unknown, update the reveals count
                    self._player_reveals[self.PIECE_KEY.index(to_piece[1])] += 1
                    # mark the piece on the board as revealed
                    self._board[to_ring][to_index] = to_piece + '!'

    def get_possible_moves(self, player):
        moves = []
        for ring in 'OIC':
            squares = self._board[ring]
            for (index, piece) in enumerate(squares):
                if piece[0] == player:
                    moves.extend([
                        ((ring, index), (r, i))
                        for (r, i) in self.get_piece_moves(ring, index)
                    ])
        return moves

    def get_piece_moves(self, ring, index):
        result = []
        player = self._board[ring][index][0]
        if player == '0':
            return result

        if ring == 'O':
            left = (index + 1) % 18
            if self._board[ring][left][0] != player:
                result.append((ring, left))
            right = (index + 17) % 18
            if self._board[ring][right][0] != player:
                result.append((ring, right))
            inside = int(math.floor(index / 2.0))
            if self._board['I'][inside][0] != player:
                result.append(('I', inside))
        elif ring == 'I':
            left = (index + 1) % 9
            if self._board[ring][left][0] != player:
                result.append((ring, left))
            right = (index + 8) % 9
            if self._board[ring][right][0] != player:
                result.append((ring, right))
            outside = int(math.floor(index * 2))
            if self._board['O'][outside][0] != player:
                result.append(('O', outside))
            outside += 1
            if self._board['O'][outside][0] != player:
                result.append(('O', outside))
            if self._board['C'][0][0] != player:
                result.append(('C', 0))
        elif ring == 'C':
            result.extend(
                [('I', i) for (i, v) in
                 enumerate(self._board['I']) if v[0] != player]
            )

        return result

    def is_match_over(self):
        counter_pieces = {'R': 1, 'P': 2, 'S': 0}
        player_counts = self._player_counts
        opponent_counts = [3 - self._captures[i] for i in range(3)]
        if sum(player_counts) == 0:
            return 1.0, 'O'
        if sum(opponent_counts) == 0:
            return 1.0, 'P'
        if self._board['C'][0] != '0':
            center = self._board['C'][0]
            if center[0] == 'P' and opponent_counts[counter_pieces[center[1]]] == 0:
                return 1.0, 'P'
            if center[0] == 'O' and center[1] != 'U' and player_counts[counter_pieces[center[1]]] == 0:
                return 1.0, 'O'
            if center[0] == 'O' and center[1] == 'U' and 0 in player_counts:
                # opponent may or may not have won!
                prob_pieces = self.get_opponent_piece_probabilities()
                # find which pieces we've lost all of
                loser_indices = [i for i in range(len(player_counts)) if player_counts[i] == 0]
                # and see which piece can win against that
                winner_indices = [(loser_index + 2) % 3 for loser_index in loser_indices]
                # and return the probability that the center has that
                prob_defeat = sum([prob_pieces[winner_index] for winner_index in winner_indices])
                return prob_defeat, 'O'
        return 0.0, None

    def get_opponent_piece_probabilities(self):
        probabilities = [0.0] * 3
        if self._opponent_counts[3] > 0:
            probabilities = [
                (3.0 - self._opponent_counts[i]) / self._opponent_counts[3] for i in range(3)
            ]
        return probabilities

    def get_hash(self):
        return ''.join(self._board['O']) \
               + ''.join(self._board['I']) \
               + ''.join(self._board['C']) \
               + '-' \
               + ''.join(str(c) for c in self._captures)

    def print_board(self):
        template = """
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
        for ring in self._board.keys():
            for index in range(len(self._board[ring]) - 1, -1, -1):
                value = self._board[ring][index]
                template = template.replace("%s%d" % (ring, index), '..' if value == '0' else value)
        print(template)
        print("Turns:", self._turns)
        print("Player Counts:", self._player_counts)
        print("Player Reveals:", self._player_reveals)
        print("Opponent Captures:", self._captures)
        print("Opponent Counts:", self._opponent_counts)
        print("Probabilities:", self.get_opponent_piece_probabilities())
