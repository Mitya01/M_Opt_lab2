import ast
import math
import time
import numpy as np
import matplotlib.pyplot as plt


def create_function(expression):
    """Разбор строки: x, pi, e, функции и операции +, -, *, /, **."""
    expression = expression.strip()
    if expression.startswith('f(x)') and '=' in expression:
        expression = expression.split('=', 1)[1].strip()
    expression = expression.replace('^', '**')

    names = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
        'tg': math.tan, 'abs': abs, 'pi': math.pi, 'e': math.e
    }
    # Проверка синтаксиса до eval: атрибуты, индексы, импорт и прочий
    # произвольный Python-код не допускаются.
    tree = ast.parse(expression, mode='eval')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
               ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError('Недопустимый элемент в формуле.')
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError('В формуле допустимы только числовые константы.')
        if isinstance(node, ast.Name):
            if node.id != 'x' and node.id not in names:
                raise ValueError('Неизвестное имя: ' + node.id)
        if isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name)
                    or not callable(names.get(node.func.id))
                    or len(node.args) != 1 or node.keywords):
                raise ValueError('Допустимы перечисленные функции с одним аргументом.')

    code = compile(tree, '<function>', 'eval')

    def f(x):
        values = names.copy()
        values['x'] = float(x)
        try:
            y = float(eval(code, {'__builtins__': {}}, values))
        except (ArithmeticError, TypeError, ValueError) as error:
            raise ValueError(f'Ошибка вычисления при x={x}: {error}') from error
        if not math.isfinite(y):
            raise ValueError(f'Неконечное значение функции при x={x}.')
        return y

    return f


def intersection(x_left, y_left, x_right, y_right, L):
    """Точка пересечения двух прямых с наклонами -L и +L."""
    width = x_right - x_left
    tolerance = 32 * np.finfo(float).eps * max(1.0, abs(y_left), abs(y_right), L * width)
    if abs(y_right - y_left) > L * width + tolerance:
        raise ValueError('L слишком мало: нарушено условие Липшица между пробами.')

    x_bottom = x_left + width / 2 + (y_left - y_right) / (2 * L)
    y_bottom = y_left / 2 + y_right / 2 - L * width / 2
    if not math.isfinite(x_bottom) or not math.isfinite(y_bottom):
        raise ValueError('Переполнение при расчёте пересечения.')
    return x_bottom, y_bottom


def broken_lines(f, a, b, L, eps, max_iter=10000):
    """Метод ломаных с явным хранением верхних и нижних вершин."""
    if not all(math.isfinite(v) for v in (a, b, L, eps)):
        raise ValueError('Параметры должны быть конечными.')
    if a >= b or L <= 0 or eps <= 0:
        raise ValueError('Требуется a < b, L > 0, eps > 0.')
    if type(max_iter) is not int or max_iter < 0:
        raise ValueError('max_iter должен быть неотрицательным целым.')

    start = time.perf_counter()
    upper_x = [float(a), float(b)]
    upper_y = [f(a), f(b)]
    tx, ty = intersection(a, upper_y[0], b, upper_y[1], L)
    lower_x = [tx]
    lower_y = [ty]

    if upper_y[0] <= upper_y[1]:
        x_best, U = upper_x[0], upper_y[0]
    else:
        x_best, U = upper_x[1], upper_y[1]

    iterations = 0
    gap_history = []
    converged = False

    while True:
        # Выбираем минимум только среди аналитически рассчитанных
        # нижних вершин. Перебора точек плотной сетки здесь нет.
        k = int(np.argmin(lower_y))
        B = lower_y[k]
        gap = max(0.0, U - B)
        gap_history.append(gap)

        if gap <= eps:
            converged = True
            status = 'Заданная точность достигнута.'
            break
        if iterations >= max_iter:
            status = 'Лимит итераций: точность не подтверждена.'
            break

        x_new = lower_x[k]
        if not upper_x[k] < x_new < upper_x[k + 1]:
            status = 'Предел машинной точности: точность не подтверждена.'
            break
        y_new = f(x_new)

        # Старую нижнюю вершину заменяем двумя новыми пересечениями.
        left_x, left_y = intersection(
            upper_x[k], upper_y[k], x_new, y_new, L)
        right_x, right_y = intersection(
            x_new, y_new, upper_x[k + 1], upper_y[k + 1], L)
        lower_x[k:k + 1] = [left_x, right_x]
        lower_y[k:k + 1] = [left_y, right_y]

        # Добавляем новую верхнюю вершину между прежними соседями.
        upper_x.insert(k + 1, x_new)
        upper_y.insert(k + 1, y_new)
        iterations += 1
        if y_new < U:
            x_best, U = x_new, y_new


    elapsed = time.perf_counter() - start
    return {
        'x_min': x_best, 'f_min': U, 'lower_bound': B, 'gap': gap,
        'iterations': iterations, 'evaluations': iterations + 2,
        'time': elapsed, 'converged': converged, 'status': status,
        'upper_x': upper_x, 'upper_y': upper_y,
        'lower_x': lower_x, 'lower_y': lower_y,
        'gap_history': gap_history
    }


def draw_result(f, a, b, eps, result, output_path='result.png'):
    # Ломаная строится из хранимых вершин, а не из значений на сетке.
    line_x, line_y = [], []
    for i in range(len(result['lower_x'])):
        line_x.extend([result['upper_x'][i], result['lower_x'][i]])
        line_y.extend([result['upper_y'][i], result['lower_y'][i]])
    line_x.append(result['upper_x'][-1])
    line_y.append(result['upper_y'][-1])

    # Эти вычисления не участвуют в поиске и не входят в его счётчик/время.
    plot_x = np.linspace(a, b, 4000)
    plot_y = [f(x) for x in plot_x]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    axes[0].plot(plot_x, plot_y, color='royalblue', label='Исходная функция')
    axes[0].plot(line_x, line_y, color='darkorange', linewidth=1.2,
                 label='Итоговая ломаная')
    axes[0].scatter(result['upper_x'], result['upper_y'], s=13,
                    color='royalblue', label='Верхние вершины', zorder=3)
    axes[0].scatter(result['lower_x'], result['lower_y'], s=13,
                    color='darkorange', marker='v', label='Нижние вершины', zorder=3)
    axes[0].scatter(result['x_min'], result['f_min'], s=150,
                    color='red', marker='*', label='Найденный минимум', zorder=5)
    axes[0].set(title='Метод ломаных', xlabel='x', ylabel='f(x)', xlim=(a, b))
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(np.maximum(result['gap_history'], 1e-16),
                     color='seagreen', label='U − B')
    axes[1].axhline(eps, color='red', linestyle='--', label=f'eps = {eps}')
    axes[1].set(title='Оценка погрешности', xlabel='Итерация', ylabel='U − B')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.show()

def read_input(filename):
    """Формат: формула; a b eps; необязательная третья строка L."""
    with open(filename, 'r', encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    # Допускаем пустые строки в конце файла.
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) not in (2, 3) or not lines[0].strip():
        raise ValueError('Нужны две строки: формула и a b eps. Третья строка L необязательна.')
    expression = lines[0].strip()
    parameters = lines[1].split()
    if len(parameters) != 3:
        raise ValueError('Во второй строке должны быть ровно три числа: a b eps.')
    a, b, eps = map(float, parameters)
    L = float(lines[2]) if len(lines) == 3 else None
    return expression, a, b, eps, L

def constant_estimation(f, a, b, n=1000, reserve=1.2):
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        raise ValueError("Нужны конечные границы a < b.")

    if not isinstance(n, int) or n < 2:
        raise ValueError("Число точек n должно быть целым и не меньше 2.")

    if not np.isfinite(reserve) or reserve <= 1:
        raise ValueError("Коэффициент запаса должен быть больше 1.")

    xs = np.linspace(a, b, n)
    ys = np.array([f(float(x)) for x in xs], dtype=float)

    if not np.all(np.isfinite(ys)):
        raise ValueError("Получены неконечные значения функции.")

    dx = np.diff(xs)
    if np.any(dx <= 0):
        raise ValueError("Точки сетки сливаются из-за машинной точности.")

    slopes = np.abs(np.diff(ys)) / dx
    L = float(np.max(slopes))

    if not np.isfinite(L):
        raise ValueError("Переполнение при вычислении наклонов.")

    return max(L, 1e-9) * reserve

if __name__ == '__main__':
    input_path = 'input.txt'
    output_path = 'result.png'

    try:
        expression, a, b, eps, L = read_input(input_path)
        if L is None:
            L = constant_estimation(f, a, b, n=1000)

        f = create_function(expression)
        result = broken_lines(f, a, b, L, eps)

        print('Функция:', expression)
        print(f'Отрезок: [{a}; {b}]; L = {L:.10f}; eps = {eps}')
        print(f"Аргумент минимума: {result['x_min']:.10f}")
        print(f"Значение функции: {result['f_min']:.10f}")
        print(f"Нижняя оценка B: {result['lower_bound']:.10f}")
        print(f"Оценка погрешности U − B: {result['gap']:.10f}")
        print(f"Итераций: {result['iterations']}")
        print(f"Вычислений функции: {result['evaluations']}")
        print(f"Время поиска без отрисовки: {result['time']:.6f} с")
        print(result['status'])
        print('Гарантия точности предполагает корректное L на всём отрезке.')

        draw_result(f, a, b, eps, result, output_path)
        print('График сохранён:', output_path)
    except (OSError, ValueError, SyntaxError, EOFError) as error:
        print('Ошибка:', error)
        raise SystemExit(1)

