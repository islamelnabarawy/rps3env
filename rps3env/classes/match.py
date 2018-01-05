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
        self._game_over = False

    @property
    def board(self):
        return self._board

    @property
    def moves(self):
        return self._moves

    @property
    def game_over(self):
        return self._game_over

    def set_board(self, pieces: list, color: PlayerColor):
        assert self._round[color.value] < 0
        assert isinstance(pieces, list) and len(pieces) == 9
        assert pieces.count(PieceType.R.value) == \
               pieces.count(PieceType.P.value) == \
               pieces.count(PieceType.S.value) == 3

        for i, v in enumerate(pieces):
            self._board[i + (9 * color.value)] = BoardPiece(PieceType(v), color)

        self._round[color.value] += 1
        self._moves.append((pieces, color))

    def make_move(self, move_from: int, move_to: int, color: PlayerColor):
        assert not self._game_over
        assert self._round[color.value] >= 0 and self._round[1 - color.value] >= 0
        assert self._round[color.value] < self._round[1 - color.value] or \
               (color == PlayerColor.Blue and self._round[color.value] == self._round[1 - color.value])
        assert move_to in valid_locations(move_from)

        from_piece = self._board[move_from]  # type: BoardPiece
        to_piece = self._board[move_to]  # type: BoardPiece

        assert from_piece is not None and from_piece.color == color

        if to_piece is None:
            # this is a move action, just swap the piece across
            self._board[move_from], self._board[move_to] = None, from_piece
            result = 0
        else:
            assert to_piece.color != color

            # both pieces get revealed regardless of the outcome
            from_piece.revealed = True
            to_piece.revealed = True

            if to_piece.piece_type == from_piece.piece_type:
                result = 0
            elif to_piece.piece_type == [PieceType.S, PieceType.R, PieceType.P][from_piece.piece_type.value - 1]:
                # challenge won, move the piece across
                self._board[move_from], self._board[move_to] = None, from_piece
                result = 1
            else:
                # challenge lost, challenger gets destroyed
                self._board[move_from] = None
                result = -1

        # check game-over conditions
        center_piece = self._board[27]
        if center_piece is not None:
            other_pieces = [p.piece_type for p in self._board if p is not None and p.color != center_piece.color]
            if other_pieces.count([PieceType.P, PieceType.S, PieceType.R][center_piece.piece_type.value - 1]) == 0:
                result += 100 if center_piece.color == color else -100
                self._game_over = True

        if not self._game_over and len([p for p in self._board if p is not None and p.color == color]) == 0:
            result -= 100
            self._game_over = True

        if not self._game_over and len([p for p in self._board if p is not None and p.color != color]) == 0:
            result += 100
            self._game_over = True

        self._round[color.value] += 1
        self._moves.append((move_from, move_to, color))

        return result, (to_piece.piece_type if to_piece is not None else None)

    def clone(self):
        other = Match()
        other._board = [BoardPiece(x.piece_type, x.color, x.revealed) if x is not None else None for x in self._board]
        other._round = self._round[:]
        other._moves = self._moves[:]
        return other

    def get_possible_moves(self):
        assert self._round[0] >= 0 and self._round[1] >= 0
        color = PlayerColor.Blue if self._round[0] <= self._round[1] else PlayerColor.Red
        moves = []
        for i, p in enumerate(self._board):
            if p is not None and p.color == color:
                moves.extend(
                    (i, x) for x in valid_locations(i) if self._board[x] is None or self._board[x].color != color
                )
        return color, moves


def valid_locations(index):
    if index < 18:
        return [(index + 1) % 18, (index + 17) % 18, 18 + index // 2]
    index -= 18
    if index < 9:
        return [18 + (index + 1) % 9, 18 + (index + 8) % 9, index * 2, index * 2 + 1, 27]
    return [18 + i for i in range(9)]
