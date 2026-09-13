import unittest

from tetris import HEIGHT, WIDTH, Tetris, rotate_cells


class TetrisTests(unittest.TestCase):
    def test_frame_dimensions_and_rgb(self):
        frame = Tetris(seed=1).frame()
        self.assertEqual(len(frame), HEIGHT)
        self.assertTrue(all(len(row) == WIDTH for row in frame))
        self.assertTrue(all(len(pixel) == 3 for row in frame for pixel in row))

    def test_piece_stays_inside_after_moves(self):
        game = Tetris(seed=2)
        for _ in range(20):
            game.move(-1)
        self.assertTrue(game._valid(game.piece))
        for _ in range(20):
            game.move(1)
        self.assertTrue(game._valid(game.piece))

    def test_four_rotations_return_to_same_shape(self):
        cells = ((0, 0), (0, 1), (1, 1), (2, 1))
        result = cells
        for _ in range(4):
            result = rotate_cells(result)
        self.assertEqual(set(result), set(cells))

    def test_full_row_clears(self):
        game = Tetris(seed=3)
        game.board[-1] = ["I"] * WIDTH
        game.piece.cells = ((0, 0),)
        game.piece.x = 0
        game.piece.y = HEIGHT - 2
        game.tick()
        self.assertEqual(game.lines, 1)
        self.assertEqual(game.score, 100)


if __name__ == "__main__":
    unittest.main()
