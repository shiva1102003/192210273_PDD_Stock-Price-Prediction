import os
import logging
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime

from helpers import apology, login_required, lookup, inr, predict_price

# Configure application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Configure session to use filesystem
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")

# Make sure Alpha Vantage API key is set
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
logging.debug(f"ALPHA_VANTAGE_API_KEY: {api_key}")
if not api_key:
    raise RuntimeError("ALPHA_VANTAGE_API_KEY not set")

@app.route("/home")
def home():
    """Public homepage"""
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""
    try:
        if session.get("user_id"):
            session.clear()
    except:
        pass

    if request.method == "POST":
        if not request.form.get("username"):
            flash("Please provide username.")
            return redirect("/login")
        elif not request.form.get("password"):
            flash("Please provide password.")
            return redirect("/login")

        rows = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            flash("Invalid username/password.")
            return redirect("/login")

        session["user_id"] = rows[0]["id"]
        return redirect("/")
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    """Log user out"""
    session.clear()
    return redirect("/home")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        if not username:
            flash("Please provide your username.")
            return redirect("/register")
        if not password:
            flash("Please provide your password.")
            return redirect("/register")
        if not confirmation:
            flash("Please confirm your password.")
            return redirect("/register")
        if password != confirmation:
            flash("Passwords do not match.")
            return redirect("/register")
        if db.execute("SELECT * FROM users WHERE username = ?", username):
            flash("Username already taken.")
            return redirect("/register")

        db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", username, generate_password_hash(password))
        flash("You have been registered successfully! Login now and start trading.")
        return redirect("/login")
    else:
        return render_template("register.html")

@app.route("/")
@login_required
def index():
    """Show portfolio of stocks"""
    rows = db.execute("SELECT * FROM stocks WHERE user_id = ?", session["user_id"])
    portfolios = []
    stock_value = 0

    for row in rows:
        stock = row["symbol"]
        stock_info = lookup(stock)
        if stock_info:
            name = stock_info["name"]
            amount = row["shares"]
            current = stock_info["price"]
            value = round(current * amount, 2)
            portfolios.append([stock, name, amount, inr(current), inr(value)])
            stock_value += value

    cash_balance = round(db.execute("SELECT cash FROM users WHERE id = ?", session["user_id"])[0]["cash"], 2)
    stock_value = round(stock_value, 2)
    grand_total = round(cash_balance + stock_value, 2)
    start_value = 10000  # Starting cash in INR
    end_value = grand_total
    profit_loss = round(end_value - start_value, 2)

    return render_template("portfolio.html", portfolios=portfolios, cash_balance=inr(cash_balance), 
                           stock_value=inr(stock_value), grand_total=inr(grand_total), 
                           start_value=inr(start_value), end_value=inr(end_value), profit_loss=inr(profit_loss))

@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote for Indian market"""
    if request.method == "POST":
        symbol = request.form.get("symbol")
        stock = lookup(symbol)
        if not stock:
            flash("Invalid stock symbol. Use NSE symbols (e.g., TCS, RELIANCE).")
            return redirect("/quote")
        stock["price"] = inr(stock["price"])  # Format as INR
        return render_template("quoted.html", stock=stock)
    else:
        return render_template("quote.html")

@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    """Buy shares of stock from Indian market"""
    if request.method == "POST":
        symbol = request.form.get("symbol")
        shares = request.form.get("shares")
        stock = lookup(symbol)

        if not symbol:
            flash("Please provide stock's symbol (e.g., TCS, RELIANCE).")
            return redirect("/buy")
        if not shares:
            flash("Please provide number of shares.")
            return redirect("/buy")

        try:
            shares = float(shares)
            if shares <= 0:
                flash("Invalid number of shares.")
                return redirect("/buy")
        except ValueError:
            flash("Invalid number of shares.")
            return redirect("/buy")

        if not stock:
            flash("Invalid stock symbol. Use NSE symbols (e.g., TCS, RELIANCE).")
            return redirect("/buy")

        price_current = stock["price"]
        value = round(price_current * shares, 2)
        cash_current = db.execute("SELECT cash FROM users WHERE id = ?", session["user_id"])[0]["cash"]
        cash_updated = round(cash_current - value, 2)

        if cash_updated < 0:
            flash("Insufficient balance for this transaction.")
            return redirect("/buy")

        symbol = stock["symbol"]
        name = stock_info["name"]
        shares = round(shares, 2)

        if db.execute("SELECT * FROM stocks WHERE user_id = ? AND symbol = ?", session["user_id"], symbol):
            shares_current = db.execute("SELECT shares FROM stocks WHERE user_id = ? AND symbol = ?", session["user_id"], symbol)[0]["shares"]
            shares_updated = round(shares_current + shares, 2)
            db.execute("UPDATE stocks SET shares = ? WHERE user_id = ? AND symbol = ?", shares_updated, session["user_id"], symbol)
        else:
            db.execute("INSERT INTO stocks (user_id, symbol, shares) VALUES (?, ?, ?)", session["user_id"], symbol, shares)

        db.execute("UPDATE users SET cash = ? WHERE id = ?", cash_updated, session["user_id"])
        db.execute("INSERT INTO transactions (user_id, symbol, name, shares, open, value, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   session["user_id"], symbol, name, shares, price_current, value, datetime.now())

        flash("Successful transaction: Bought!")
        return redirect("/")
    else:
        return render_template("buy.html")

@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock from Indian market"""
    if request.method == "POST":
        symbol = request.form.get("symbol")
        shares = request.form.get("shares")
        stock = lookup(symbol)

        if not symbol:
            flash("Please provide stock's symbol (e.g., TCS, RELIANCE).")
            return redirect("/sell")
        if not shares:
            flash("Please provide number of shares.")
            return redirect("/sell")

        try:
            shares = float(shares)
            if shares <= 0:
                flash("Invalid number of shares.")
                return redirect("/sell")
        except ValueError:
            flash("Invalid number of shares.")
            return redirect("/sell")

        if not stock:
            flash("Invalid stock symbol. Use NSE symbols (e.g., TCS, RELIANCE).")
            return redirect("/sell")

        symbol_current = db.execute("SELECT symbol FROM stocks WHERE user_id = ?", session["user_id"])
        match = any(item["symbol"] == symbol for item in symbol_current)

        if not match:
            flash("You do not own this stock.")
            return redirect("/sell")

        shares_current = db.execute("SELECT shares FROM stocks WHERE user_id = ? AND symbol = ?", session["user_id"], symbol)[0]["shares"]
        if shares > shares_current:
            flash("You do not own enough shares.")
            return redirect("/sell")

        symbol = stock["symbol"]
        name = stock["name"]
        shares_updated = round(shares_current - shares, 2)

        if shares_updated == 0:
            db.execute("DELETE FROM stocks WHERE user_id = ? AND symbol = ?", session["user_id"], symbol)
        else:
            db.execute("UPDATE stocks SET shares = ? WHERE user_id = ? AND symbol = ?", shares_updated, session["user_id"], symbol)

        price_current = stock["price"]
        value = round(price_current * shares, 2)
        cash_current = db.execute("SELECT cash FROM users WHERE id = ?", session["user_id"])[0]["cash"]
        cash_updated = round(cash_current + value, 2)
        db.execute("UPDATE users SET cash = ? WHERE id = ?", cash_updated, session["user_id"])

        db.execute("INSERT INTO transactions (user_id, symbol, name, shares, close, value, date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   session["user_id"], symbol, name, shares, price_current, value, datetime.now())

        flash("Successful transaction: Sold!")
        return redirect("/")
    else:
        rows = db.execute("SELECT symbol, shares FROM stocks WHERE user_id = ?", session["user_id"])
        return render_template("sell.html", rows=rows)

@app.route("/history")
@login_required
def history():
    """Show history of transactions"""
    rows = db.execute("SELECT * FROM transactions WHERE user_id = ?", session["user_id"])
    transactions = []
    total_buy, total_sell = 0, 0

    for row in rows:
        stock_info = lookup(row["symbol"])
        if stock_info:
            open_price = inr(row["open"]) if row["open"] else None
            close_price = inr(row["close"]) if row["close"] else None
            value = inr(row["value"])
            transactions.append([row["symbol"], stock_info["name"], row["shares"], open_price, close_price, value, row["date"]])
            if row["open"]:
                total_buy += row["value"]
            if row["close"]:
                total_sell += row["value"]

    total_buy = inr(round(total_buy, 2))
    total_sell = inr(round(total_sell, 2))
    return render_template("history.html", transactions=transactions, total_buy=total_buy, total_sell=total_sell)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """Predict next day's stock price and display graph"""
    if request.method == "POST":
        symbol = request.form.get("symbol")
        if not symbol:
            flash("Please provide a stock symbol (e.g., TCS, RELIANCE).")
            return redirect("/predict")

        prediction_data = predict_price(symbol)
        if not prediction_data:
            flash("Unable to predict price for this stock. Ensure it’s an NSE symbol (e.g., TCS, RELIANCE).")
            return redirect("/predict")

        predicted_price = prediction_data["predicted_price"]
        historical_dates = prediction_data["dates"]
        chart_prices = prediction_data["chart_prices"]

        current_price_raw = lookup(symbol)
        current_price = inr(current_price_raw["price"]) if current_price_raw else "N/A"
        predicted_price_formatted = inr(predicted_price)

        next_day = "Next Day"
        chart_dates = historical_dates + [next_day]

        return render_template("predict.html", symbol=symbol, current_price=current_price, 
                               predicted_price=predicted_price_formatted, chart_dates=chart_dates, 
                               chart_prices=chart_prices)
    else:
        return render_template("predict.html")

def errorhandler(e):
    """Handle errors"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return apology(e.name, e.code)

# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)