import sys
import numpy as np
import sympy as sp
from sympy import symbols, lambdify, integrate, sympify
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QGridLayout,
    QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class PotentialFieldVisualization(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Моделирование потенциального поля')

        self.layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.fx_input = QLineEdit()
        self.grid_layout.addWidget(QLabel('Компонента силы Fₓ(x, y):'), 0, 0)
        self.grid_layout.addWidget(self.fx_input, 0, 1)

        self.fy_input = QLineEdit()
        self.grid_layout.addWidget(QLabel('Компонента силы Fᵧ(x, y):'), 1, 0)
        self.grid_layout.addWidget(self.fy_input, 1, 1)

        self.layout.addLayout(self.grid_layout)

        self.canvas = FigureCanvas(plt.figure())
        self.ax = self.canvas.figure.add_subplot(111)
        self.layout.addWidget(self.canvas)

        self.compute_button = QPushButton('Вычислить и визуализировать')
        self.compute_button.clicked.connect(self.compute_and_visualize)
        self.layout.addWidget(self.compute_button)

        self.setLayout(self.layout)

    def show_error(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle('Ошибка')
        msg_box.setText(message)
        msg_box.exec_()

    def preprocess_input(self, expr):
        expr = expr.replace('−', '-')
        return expr

    def validate_expression(self, expr, allowed_symbols):
        try:
            parsed_expr = sympify(expr)
        except Exception as e:
            raise ValueError(f"Некорректное выражение: {expr}. Деталь: {str(e)}")

        free_symbols = parsed_expr.free_symbols
        invalid_symbols = free_symbols - allowed_symbols

        if invalid_symbols:
            invalid_symbols_str = ', '.join([str(s) for s in invalid_symbols])
            raise ValueError(f"Использованы неопределенные переменные: {invalid_symbols_str}. Допустимы только x и y.")

        return parsed_expr

    def compute_and_visualize(self):
        try:
            fx_expr_raw = self.fx_input.text()
            fy_expr_raw = self.fy_input.text()

            fx_expr = self.preprocess_input(fx_expr_raw)
            fy_expr = self.preprocess_input(fy_expr_raw)

            x, y = symbols('x y')
            allowed_symbols = {x, y}

            Fx = self.validate_expression(fx_expr, allowed_symbols)
            Fy = self.validate_expression(fy_expr, allowed_symbols)

            Ux = -integrate(Fx, x)
            Uy = -integrate(Fy, y)
            U = Ux + Uy

            Ux_check = sp.diff(U, x)
            Uy_check = sp.diff(U, y)

            if not sp.simplify(Ux_check + Fx) == 0:
                raise ValueError("Интегрирование по x дало некорректный результат.")
            if not sp.simplify(Uy_check + Fy) == 0:
                raise ValueError("Интегрирование по y дало некорректный результат.")

            U = U.simplify()

            U_func = lambdify((x, y), U, modules=['numpy'])

            X, Y = np.meshgrid(
                np.linspace(-10, 10, 400),
                np.linspace(-10, 10, 400),
            )

            U_vals = U_func(X, Y)

            if np.iscomplexobj(U_vals):
                raise ValueError("Потенциальная энергия имеет комплексные значения. Проверьте корректность силового поля.")

            if not np.all(np.isfinite(U_vals)):
                raise ValueError("Потенциальная энергия содержит бесконечные или неопределенные значения. Проверьте корректность силового поля.")

            self.ax.clear()
            contour = self.ax.contourf(X, Y, U_vals, levels=750, cmap='viridis')
            self.ax.set_title('Потенциальная энергия U(x, y)')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.canvas.draw()

        except ValueError as ve:
            self.show_error(f'Ошибка: {str(ve)}')
        except Exception as e:
            self.show_error(f'Неизвестная ошибка при вычислении: {str(e)}')


def main():
    app = QApplication(sys.argv)
    window = PotentialFieldVisualization()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
