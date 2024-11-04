from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

import os
os.environ['FLASK_SKIP_DOTENV'] = '1'

def generate_plots(N, mu, sigma2, S):

    # STEP 1
    # Generate a random dataset X of size N with values between 0 and 1
    # and a random dataset Y with normal additive error (mean mu, variance sigma^2)
    X = np.random.uniform(0, 1, N)
    Y = np.random.normal(mu, np.sqrt(sigma2), N)

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure()
    plt.scatter(X, Y, color='gray', alpha=0.5)
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Regression Line: Y = {slope:.2f}X + {intercept:.2f}")
    plt.savefig("static/plot1.png")
    plt.close()
    
    # Initialize empty lists for slopes and intercepts
    slopes = []
    intercepts = []

    # Run a loop S times to generate datasets and calculate slopes and intercepts
    for _ in range(S):
        # Generate random X values with size N between 0 and 1
        X_sim = np.random.uniform(0, 1, N)

        # Generate Y values with normal additive error (mean mu, variance sigma^2)
        Y_sim = np.random.normal(mu, np.sqrt(sigma2), N)

        # Fit a linear regression model to X_sim and Y_sim
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)

        # Append the slope and intercept of the model to slopes and intercepts lists
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("static/plot2.png")
    plt.close()

    # Calculate proportions of more extreme slopes and intercepts
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return "static/plot1.png", "static/plot2.png", slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
