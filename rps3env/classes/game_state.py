"""
   Copyright 2019 Islam Elnabarawy

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
from copy import deepcopy

from rps3env.classes import PlayerColor, BoardPiece, PieceType

__author__ = "Islam Elnabarawy"


class GameState(object):
    """
    Game state representation for search algorithms
    """

    def __init__(self):
        self._board = [None] * 28
        self._round = -2
        self._player = PlayerColor.Blue
        self._moves = []
        self._game_over = False

    def get_possible_actions(self):
        """
        Get a list of all possible actions in the current state.

        :return: A list of Actions that are legal next actions in this state
        :rtype: list[Action]
        """
        if self._round < 0:
            # initial board setup
            return [Action(self._player, list(x)) for x in itertools.permutations([1, 2, 3] * 3)]
        moves = []
        for i, p in enumerate(self._board):
            if p is not None and p.color == self._player:
                moves.extend(
                    (i, x) for x in self.valid_locations(i)
                    if self._board[x] is None or self._board[x].color != self._player
                )
        return [Action(self._player, move) for move in moves]

    @staticmethod
    def valid_locations(index):
        if index < 18:
            return [(index + 1) % 18, (index + 17) % 18, 18 + index // 2]
        index -= 18
        if index < 9:
            return [18 + (index + 1) % 9, 18 + (index + 8) % 9, index * 2, index * 2 + 1, 27]
        return [18 + i for i in range(9)]

    def take_action(self, action):
        """
        Apply the given Action to a copy of the current state

        :type action: Action
        :rtype: GameState
        """
        assert not self._game_over
        assert action in self.get_possible_actions()

        new_state = deepcopy(self)

        if self._round < 0:
            for i, v in enumerate(action.move):
                new_state._board[i + (9 * self._player.value)] = BoardPiece(PieceType(v), self._player)
        else:
            move_from, move_to = action.move
            from_piece = new_state._board[move_from]
            to_piece = new_state._board[move_to]

            if to_piece is None:
                # this is a move action, just swap the piece across
                new_state._board[move_from], new_state._board[move_to] = None, from_piece
            else:
                # both pieces get revealed regardless of the outcome
                from_piece.revealed = True
                to_piece.revealed = True

                if to_piece.piece_type == from_piece.piece_type:
                    # it's a tie
                    pass
                elif to_piece.piece_type == [PieceType.S, PieceType.R, PieceType.P][from_piece.piece_type.value - 1]:
                    # challenge won, move the piece across
                    new_state._board[move_from], new_state._board[move_to] = None, from_piece
                else:
                    # challenge lost, challenger gets destroyed
                    new_state._board[move_from] = None

            # check game-over conditions
            center_piece = new_state._board[27]
            if center_piece is not None:
                other_pieces = [p.piece_type for p in new_state._board if
                                p is not None and p.color != center_piece.color]
                if other_pieces.count([PieceType.P, PieceType.S, PieceType.R][center_piece.piece_type.value - 1]) == 0:
                    new_state._game_over = True

            if not new_state._game_over and len(
                    [p for p in new_state._board if p is not None and p.color == self._player]) == 0:
                new_state._game_over = True

            if not new_state._game_over and len(
                    [p for p in new_state._board if p is not None and p.color != self._player]) == 0:
                new_state._game_over = True

        new_state._player = PlayerColor(1 - self._player.value)
        new_state._moves.append(action)
        new_state._round += 1

        return new_state

    def is_terminal(self):
        """
        Get whether the current state is a terminal state

        :return: A boolean indicating whether this is a terminal state
        :rtype: bool
        """
        return self._game_over

    def get_reward(self):
        """
        Get the final reward for a terminal state

        :return: The final reward for this state, assuming it is a terminal state
        :rtype: int
        """
        # only needed for terminal states
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        return 'GameState(round={}, player={})'.format(self._round, self._player)


class Action(object):
    def __init__(self, player, move):
        self.player = player
        self.move = move

    def __eq__(self, other):
        """
        :type other: Action
        """
        return self.__class__ == other.__class__ \
               and self.player == other.player \
               and self.move == other.move

    def __hash__(self):
        return hash((self.player, self.move))

    def __repr__(self) -> str:
        return 'Action(\n\t{}\n)'.format(',\n\t'.join([
            '{}={}'.format(k, self.__dict__[k])
            for k in sorted(self.__dict__.keys())
        ]))

    def __str__(self) -> str:
        return str((self.player, self.move))
