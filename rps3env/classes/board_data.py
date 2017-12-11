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
from enum import Enum

__author__ = 'Islam Elnabarawy'


class PieceType(Enum):
    N = -1
    U = 0
    R = 1
    P = 2
    S = 3


class BoardPiece(object):
    def __init__(self, piece_type: PieceType, player_owned: bool, revealed: bool = False) -> None:
        super().__init__()
        self._piece_type = piece_type
        self._player_owned = player_owned
        self._revealed = revealed

    def to_str(self, hidden=True):
        if self._player_owned:
            return 'P' + self._piece_type.name
        return 'O' + (self._piece_type.name if self.revealed or not hidden else 'U')

    @property
    def piece_type(self) -> PieceType:
        return self._piece_type

    @property
    def player_owned(self) -> bool:
        return self._player_owned

    @property
    def revealed(self) -> bool:
        return self._revealed

    @revealed.setter
    def revealed(self, value):
        self._revealed = value


class BoardLocation(object):
    def __init__(self, ring: str, index: int, piece: BoardPiece = None) -> None:
        super().__init__()
        self._ring = ring
        self._index = index
        self._piece = piece

    @property
    def ring(self) -> str:
        return self._ring

    @property
    def index(self) -> int:
        return self._index

    @property
    def piece(self) -> BoardPiece:
        return self._piece

    @piece.setter
    def piece(self, value: BoardPiece) -> None:
        self._piece = value
