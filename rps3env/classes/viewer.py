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
import os

from rps3env.classes import BoardPiece, PlayerColor

__author__ = 'Islam Elnabarawy'

BOARD_POSITIONS = [
    (557, 250), (518, 156), (455, 89), (371, 49), (279, 38), (191, 62), (119, 110), (67, 180), (42, 259),
    (42, 347), (78, 440), (147, 512), (228, 550), (321, 559), (408, 536), (478, 489), (531, 420), (558, 339),
    (463, 234), (380, 141), (255, 128), (158, 191), (122, 305), (172, 421), (283, 475), (400, 444), (467, 355),
    (300, 300)
]


class Viewer(object):
    """Viewer for rendering a graphical representation of game states"""

    def __init__(self, width, height, board_offset_x=50, board_offset_y=50, bg_height=600, bg_width=600) -> None:
        import pyglet
        from pyglet import gl

        self.width = width
        self.height = height

        self.board_offset_x = board_offset_x
        self.board_offset_y = board_offset_y

        self.bg_height = bg_height
        self.bg_width = bg_width

        self._bg = pyglet.image.load(os.path.join(os.path.dirname(__file__), '../assets/board.png'))
        self._window = pyglet.window.Window(width=width, height=height)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def _draw_piece(self, index: int, piece: BoardPiece):
        x, y = BOARD_POSITIONS[index]
        color = (0, 0, 255, 255) if piece.color == PlayerColor.Blue else (255, 0, 0, 255)
        import pyglet
        pyglet.text.Label(
            ('{}!' if piece.revealed else '{}').format(piece.piece_type.name),
            font_name='Arial', font_size=28, anchor_x='center', anchor_y='center',
            x=x + self.board_offset_x, y=y + self.board_offset_y, color=color
        ).draw()

    def render(self, board, match_result=None, return_rgb_array=False):
        import pyglet
        from pyglet import gl
        import numpy as np

        gl.glClearColor(1, 1, 1, 1)
        self._window.clear()
        self._window.switch_to()
        self._window.dispatch_events()

        self._bg.blit(self.board_offset_x, self.board_offset_y, width=self.bg_width, height=self.bg_height)

        for i, p in enumerate(board):
            if p is not None:
                self._draw_piece(i, p)

        if match_result is not None:
            txt = "Game Over! {} won.".format('Player' if match_result > 0 else 'Opponent')
            label = pyglet.text.Label(
                txt, font_name='Arial', font_size=32, anchor_x='center', anchor_y='center',
                x=350, y=30, color=(0, 0, 0, 255)
            )
            label.draw()

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            arr = np.fromstring(buffer.get_image_data().data, dtype=np.uint8, sep='')
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self._window.flip()

        return arr

    def close(self):
        self._window.close()
