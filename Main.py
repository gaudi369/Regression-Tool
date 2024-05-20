import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog

# Regression functions
def linear_regression(x, y):
    model = LinearRegression()
    x = x.reshape(-1, 1)
    model.fit(x, y)
    y_pred = model.predict(x)
    return y_pred, model.coef_[0], model.intercept_

def polynomial_regression(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    x = x.reshape(-1, 1)
    model.fit(x, y)
    y_pred = model.predict(x)
    return y_pred, model

def logistic_growth(x, K, r, x0):
    return K / (1 + np.exp(-r * (x - x0)))

def nonlinear_regression(x, y):
    initial_guesses = [max(y), 1, np.median(x)]
    popt, _ = curve_fit(logistic_growth, x, y, p0=initial_guesses, maxfev=2000)
    y_pred = logistic_growth(x, *popt)
    return y_pred, popt

# GUI application
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prey-Predator Population Analysis")
        self.geometry("800x800")

        self.steps = None
        self.prey_population = None
        self.predator_population = None

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self, text="Load a CSV file to start")
        self.label.pack(pady=10)

        self.button_load_csv = tk.Button(self, text="Load CSV", command=self.load_csv)
        self.button_load_csv.pack(pady=10)

        self.window_var = tk.StringVar(value="full")
        self.radio_full = tk.Radiobutton(self, text="Full", variable=self.window_var, value="full", command=self.toggle_window_entries)
        self.radio_windowed = tk.Radiobutton(self, text="Windowed", variable=self.window_var, value="windowed", command=self.toggle_window_entries)
        self.radio_full.pack()
        self.radio_windowed.pack()

        self.label_start = tk.Label(self, text="Start of range:")
        self.label_start.pack(pady=5)
        self.entry_start = tk.Entry(self)
        self.entry_start.pack(pady=5)

        self.label_end = tk.Label(self, text="End of range:")
        self.label_end.pack(pady=5)
        self.entry_end = tk.Entry(self)
        self.entry_end.pack(pady=5)

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

        self.toggle_window_entries()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            data = pd.read_csv(file_path)
            self.steps = data.iloc[:, 0].values
            self.prey_population = data.iloc[:, 1].values
            self.predator_population = data.iloc[:, 2].values
            self.plot_data()

    def toggle_window_entries(self):
        state = 'normal' if self.window_var.get() == 'windowed' else 'disabled'
        self.entry_start.config(state=state)
        self.entry_end.config(state=state)

    def get_windowed_data(self):
        if self.window_var.get() == 'windowed':
            try:
                start = int(self.entry_start.get())
                end = int(self.entry_end.get())
                mask = (self.steps >= start) & (self.steps <= end)
                return self.steps[mask], self.prey_population[mask], self.predator_population[mask]
            except ValueError:
                tk.messagebox.showerror("Invalid input", "Please enter valid start and end values.")
                return self.steps, self.prey_population, self.predator_population
        else:
            return self.steps, self.prey_population, self.predator_population

    def plot_data(self):
        if self.steps is None or self.prey_population is None or self.predator_population is None:
            return

        self.ax.clear()

        steps, prey_population, predator_population = self.get_windowed_data()

        # Plotting populations
        self.ax.plot(steps, prey_population, label='Prey Population', color='blue')
        self.ax.plot(steps, predator_population, label='Predator Population', color='green')

        # Regression Analysis
        x = steps
        y_prey = np.array(prey_population)
        y_predator = np.array(predator_population)

        # Linear Regression
        if self.check_var_lin.get():
            y_pred_prey_lin, coef_prey_lin, intercept_prey_lin = linear_regression(x, y_prey)
            y_pred_predator_lin, coef_predator_lin, intercept_predator_lin = linear_regression(x, y_predator)
            self.ax.plot(x, y_pred_prey_lin, label=f'Prey Linear: y={coef_prey_lin:.2f}x+{intercept_prey_lin:.2f}', linestyle='--', color='red')
            self.ax.plot(x, y_pred_predator_lin, label=f'Predator Linear: y={coef_predator_lin:.2f}x+{intercept_predator_lin:.2f}', linestyle='--', color='red')

        # Polynomial Regression
        if self.check_var_poly.get():
            degree = 3
            y_pred_prey_poly, model_prey_poly = polynomial_regression(x, y_prey, degree)
            y_pred_predator_poly, model_predator_poly = polynomial_regression(x, y_predator, degree)
            coef_prey_poly = model_prey_poly.named_steps['linearregression'].coef_
            intercept_prey_poly = model_prey_poly.named_steps['linearregression'].intercept_
            coef_predator_poly = model_predator_poly.named_steps['linearregression'].coef_
            intercept_predator_poly = model_predator_poly.named_steps['linearregression'].intercept_
            self.ax.plot(x, y_pred_prey_poly, label=f'Prey Polynomial: y={coef_prey_poly[3]:.2f}x^3+{coef_prey_poly[2]:.2f}x^2+{coef_prey_poly[1]:.2f}x+{intercept_prey_poly:.2f}', linestyle='--', color='green')
            self.ax.plot(x, y_pred_predator_poly, label=f'Predator Polynomial: y={coef_predator_poly[3]:.2f}x^3+{coef_predator_poly[2]:.2f}x^2+{coef_predator_poly[1]:.2f}x+{intercept_predator_poly:.2f}', linestyle='--', color='green')

        # Non-linear Regression
        if self.check_var_nonlin.get():
            y_pred_prey_nonlin, popt_prey_nonlin = nonlinear_regression(x, y_prey)
            y_pred_predator_nonlin, popt_predator_nonlin = nonlinear_regression(x, y_predator)
            self.ax.plot(x, y_pred_prey_nonlin, label=f'Prey Non-linear: K={popt_prey_nonlin[0]:.2f}, r={popt_prey_nonlin[1]:.2f}, x0={popt_prey_nonlin[2]:.2f}', linestyle='--', color='orange')
            self.ax.plot(x, y_pred_predator_nonlin, label=f'Predator Non-linear: K={popt_predator_nonlin[0]:.2f}, r={popt_predator_nonlin[1]:.2f}, x0={popt_predator_nonlin[2]:.2f}', linestyle='--', color='orange')

        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Population')
        self.ax.legend()

        self.canvas.draw()

# Run the application
if __name__ == "__main__":
    app = Application()
    app.mainloop()
