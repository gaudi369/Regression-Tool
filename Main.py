import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog

# Regression functions
def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    nrmse = rmse / np.mean(y)
    return y_pred, model.coef_, model.intercept_, nrmse

def polynomial_regression(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x, y)
    y_pred = model.predict(x)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    nrmse = rmse / np.mean(y)
    return y_pred, model, nrmse

def logistic_growth(x, K, r, x0):
    return K / (1 + np.exp(-r * (x - x0)))

def nonlinear_regression(x, y):
    initial_guesses = [max(y), 1, np.median(x)]
    popt, _ = curve_fit(logistic_growth, x.flatten(), y, p0=initial_guesses, maxfev=5000)
    y_pred = logistic_growth(x, *popt)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    nrmse = rmse / np.mean(y)
    return y_pred, popt, nrmse

# GUI application
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Population Analysis")
        self.geometry("800x800")

        self.data = None
        self.num_dimensions = 0

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self, text="Load a CSV file to start")
        self.label.pack(pady=10)

        self.button_load_data = tk.Button(self, text="Load CSV", command=self.load_data)
        self.button_load_data.pack(pady=10)

        self.label_range = tk.Label(self, text="Range (lower bound, upper bound):")
        self.label_range.pack(pady=5)
        self.entry_range = tk.Entry(self)
        self.entry_range.pack(pady=5)

        self.label_dimensions = tk.Label(self, text="Dimensions to plot (e.g., 1,2,3):")
        self.label_dimensions.pack(pady=5)
        self.entry_dimensions = tk.Entry(self)
        self.entry_dimensions.pack(pady=5)

        self.check_var_lin = tk.BooleanVar(value=True)
        self.check_var_poly = tk.BooleanVar(value=True)
        self.check_var_nonlin = tk.BooleanVar(value=True)

        self.check_lin = tk.Checkbutton(self, text="Linear Regression", variable=self.check_var_lin, command=self.plot_data)
        self.check_poly = tk.Checkbutton(self, text="Polynomial Regression", variable=self.check_var_poly, command=self.plot_data)
        self.check_nonlin = tk.Checkbutton(self, text="Non-linear Regression", variable=self.check_var_nonlin, command=self.plot_data)

        self.check_lin.pack()
        self.check_poly.pack()
        self.check_nonlin.pack()

        self.button_plot = tk.Button(self, text="Plot", command=self.plot_data)
        self.button_plot.pack(pady=20)

        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            data = pd.read_csv(file_path)
            self.data = data
            self.num_dimensions = data.shape[1] - 1  # Assuming the first column is the independent variable
            self.plot_data()

    def get_windowed_data(self):
        range_input = self.entry_range.get()
        if range_input:
            try:
                lower, upper = map(int, range_input.split(','))
                mask = (self.data.iloc[:, 0] >= lower) & (self.data.iloc[:, 0] <= upper)
                windowed_data = self.data[mask].copy()
                windowed_data.iloc[:, 0] = windowed_data.iloc[:, 0] - lower
                return windowed_data
            except ValueError:
                tk.messagebox.showerror("Invalid input", "Please enter valid lower and upper bounds.")
                return self.data
        else:
            return self.data

    def get_selected_dimensions(self):
        dims_input = self.entry_dimensions.get()
        if dims_input:
            try:
                dims = list(map(int, dims_input.split(',')))
                return [dim for dim in dims if 1 <= dim <= self.num_dimensions]
            except ValueError:
                tk.messagebox.showerror("Invalid input", "Please enter valid dimension numbers.")
                return list(range(1, self.num_dimensions + 1))
        else:
            return list(range(1, self.num_dimensions + 1))

    def plot_data(self):
        if self.data is None:
            return

        self.ax.clear()

        data = self.get_windowed_data()
        x = data.iloc[:, 0].values.reshape(-1, 1)

        selected_dimensions = self.get_selected_dimensions()

        for i in selected_dimensions:
            y = data.iloc[:, i].values

            # Plotting the data
            self.ax.plot(x, y, label=f'Dimension {i}', color=plt.cm.tab10(i - 1))

            # Linear Regression
            if self.check_var_lin.get():
                y_pred_lin, coef_lin, intercept_lin, nrmse_lin = linear_regression(x, y)
                equation_lin = f'$y = {coef_lin[0]:.2f}x + {intercept_lin:.2f}, NRMSE = {nrmse_lin:.2f}$'
                self.ax.plot(x, y_pred_lin, label=f'Linear: {equation_lin}', linestyle='--', color=plt.cm.tab10(i - 1))

            # Polynomial Regression
            if self.check_var_poly.get():
                degree = 3
                y_pred_poly, model_poly, nrmse_poly = polynomial_regression(x, y, degree)
                coefs_poly = model_poly.named_steps['linearregression'].coef_.ravel()
                intercept_poly = model_poly.named_steps['linearregression'].intercept_
                equation_poly = f'$y = {intercept_poly:.2f} '
                for d in range(1, degree + 1):
                    equation_poly += f'+ {coefs_poly[d]:.2f}x^{d} '
                equation_poly += f', NRMSE = {nrmse_poly:.2f}$'
                self.ax.plot(x, y_pred_poly, label=f'Poly: {equation_poly}', linestyle='--', color=plt.cm.tab10(i - 1))

            # Non-linear Regression
            if self.check_var_nonlin.get():
                y_pred_nonlin, popt_nonlin, nrmse_nonlin = nonlinear_regression(x, y)
                equation_nonlin = f'$K = {popt_nonlin[0]:.2f}, r = {popt_nonlin[1]:.2f}, x0 = {popt_nonlin[2]:.2f}, NRMSE = {nrmse_nonlin:.2f}$'
                self.ax.plot(x, y_pred_nonlin, label=f'Non-lin: {equation_nonlin}', linestyle='--', color=plt.cm.tab10(i - 1))

        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Value')
        self.ax.legend()

        self.canvas.draw()

# Run the application
if __name__ == "__main__":
    app = Application()
    app.mainloop()
