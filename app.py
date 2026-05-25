from flask import Flask, render_template, request, jsonify
import numpy as np
from fractions import Fraction

app = Flask(__name__)

def gauss_jordan(matrix):
    """Perform Gauss-Jordan elimination with step tracking."""
    m = [[Fraction(x).limit_denominator(1000) for x in row] for row in matrix]
    rows, cols = len(m), len(m[0])

    steps = [{
        "description": "Initial augmented matrix",
        "matrix": format_matrix(m)
    }]

    pivot_row = 0

    for col in range(cols - 1):

        if pivot_row >= rows:
            break

        # Find pivot
        max_row = max(
            range(pivot_row, rows),
            key=lambda r: abs(m[r][col])
        )

        if m[max_row][col] == 0:
            continue

        # Swap rows
        if max_row != pivot_row:
            m[pivot_row], m[max_row] = m[max_row], m[pivot_row]

            steps.append({
                "description": f"R{pivot_row+1} ↔ R{max_row+1}",
                "matrix": format_matrix(m)
            })

        # Make pivot = 1
        scale = m[pivot_row][col]

        if scale != 1:
            m[pivot_row] = [x / scale for x in m[pivot_row]]

            steps.append({
                "description": f"R{pivot_row+1} ← (1/{scale})·R{pivot_row+1}",
                "matrix": format_matrix(m)
            })

        # Eliminate other rows
        for r in range(rows):

            if r != pivot_row and m[r][col] != 0:

                factor = m[r][col]

                m[r] = [
                    m[r][j] - factor * m[pivot_row][j]
                    for j in range(cols)
                ]

                steps.append({
                    "description": f"R{r+1} ← R{r+1} - ({factor})·R{pivot_row+1}",
                    "matrix": format_matrix(m)
                })

        pivot_row += 1

    return steps


def format_matrix(m):
    return [[str(x) for x in row] for row in m]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():

    data = request.json
    matrix = data.get('matrix', [])

    steps = gauss_jordan(matrix)

    return jsonify({
        "steps": steps
    })


if __name__ == '__main__':
    app.run(debug=True)