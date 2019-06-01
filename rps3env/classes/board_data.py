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
from enum import Enum

__author__ = 'Islam Elnabarawy'


class PlayerColor(Enum):
    Blue = 0
    Red = 1


class PieceType(Enum):
    N = -1
    U = 0
    R = 1
    P = 2
    S = 3


class BoardPiece(object):
    def __init__(self, piece_type: PieceType, color: PlayerColor, revealed: bool = False) -> None:
        super().__init__()
        self._piece_type = piece_type
        self._color = color
        self._revealed = revealed

    def to_str(self, player_color, hidden=True):
        if self._color == player_color:
            return 'P' + self._piece_type.name
        return 'O' + (self._piece_type.name if self.revealed or not hidden else 'U')

    @property
    def piece_type(self) -> PieceType:
        return self._piece_type

    @property
    def color(self) -> PlayerColor:
        return self._color

    @property
    def revealed(self) -> bool:
        return self._revealed

    @revealed.setter
    def revealed(self, value):
        self._revealed = value

    def __str__(self):
        return '{}{}{}'.format(self.color.name[0], self.piece_type.name, '!' if self.revealed else '')

    def __repr__(self) -> str:
        return 'BoardPiece(type={}, color={}, revealed={})'.format(self.piece_type.name, self.color.name, self.revealed)
