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

from rps3env.classes import BoardPiece, PieceType, PlayerColor

__author__ = 'Islam Elnabarawy'


class Match(object):

    def __init__(self) -> None:
        self._board = [None] * 28
        self._round = [-1, -1]
        self._moves = []

    @property
    def board(self):
        return self._board

    def set_board(self, pieces: list, color: PlayerColor):
        assert self._round[color.value] < 0
        assert isinstance(pieces, list) and len(pieces) == 9
        assert pieces.count(PieceType.R.value) == \
               pieces.count(PieceType.P.value) == \
               pieces.count(PieceType.S.value) == 3

        for i, v in enumerate(pieces):
            self._board[i + (9 * color.value)] = BoardPiece(PieceType(v), color)

        self._round[color.value] += 1
        self._moves.append((color, pieces))

    def make_move(self, move_from: int, move_to: int, color: PlayerColor):
        assert self._round[color.value] >= 0 and self._round[1 - color.value] >= 0
        assert self._round[color.value] < self._round[1 - color.value] or \
               (color == PlayerColor.Blue and self._round[color.value] == self._round[1 - color.value])
        assert move_to in valid_locations(move_from)

        from_piece = self._board[move_from]
        to_piece = self._board[move_to]
        result = None

        assert from_piece is not None and from_piece.color == color

        if to_piece is None:
            # this is a move action, just swap the piece across
            self._board[move_from], self._board[move_to] = None, from_piece
            result = (0, None)

        self._round[color.value] += 1
        self._moves.append((color, move_from, move_to))

        return result


def valid_locations(index):
    if index < 18:
        return [(index + 1) % 18, (index + 17) % 18, 18 + index // 2]
    index -= 18
    if index < 9:
        return [18 + (index + 1) % 9, 18 + (index + 8) % 9, index * 2, index * 2 + 1, 27]
    return [18 + i for i in range(9)]
