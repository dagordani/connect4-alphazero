// Board helpers. The server sends the board in absolute colours:
// board[row][col], row 0 = top, +1 = red (moves first), -1 = yellow, 0 = empty.

export const ROWS = 6;
export const COLS = 7;
export const COLOR = { 1: 'red', [-1]: 'yellow' };

// The four cells of a completed line, as "row,col" keys, or an empty set.
export function winningCells(board) {
  const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const piece = board[r][c];
      if (!piece) continue;
      for (const [dr, dc] of directions) {
        const cells = [0, 1, 2, 3].map((i) => [r + dr * i, c + dc * i]);
        const complete = cells.every(
          ([rr, cc]) => rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && board[rr][cc] === piece,
        );
        if (complete) return new Set(cells.map(([rr, cc]) => `${rr},${cc}`));
      }
    }
  }
  return new Set();
}

// Row where the last move in `column` landed: the topmost filled cell.
export function topRow(board, column) {
  for (let r = 0; r < ROWS; r++) if (board[r][column] !== 0) return r;
  return -1;
}

export const capitalize = (s) => s[0].toUpperCase() + s.slice(1);
