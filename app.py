from flask import Flask, render_template, url_for, request, redirect, flash, send_file, jsonify, session
import sqlite3
import json
import csv
import datetime
import nltk
from db import init_db, view_portfolio, visualise_portfolio, load_csv_data, insert_data_to_db, fetch_stock_data
from werkzeug.utils import secure_filename
import os
import yfinance as yf
import matplotlib.pyplot as plt
import hashlib
import re
import io
import base64
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests


app = Flask(__name__)
app.secret_key = "your_secret_key"
# Define the folder for uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}



conn = init_db()
usa_large_companies, stock_info, nasdaq_data, sp_data = load_csv_data()
insert_data_to_db(conn, stock_info)

# Homepage -----------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def homepage():
    innit_db()
    if request.method == 'POST':
        return redirect(url_for('homepage'))
    return render_template('homepage.html')

# Register and Login ---------------------------------------------------------------------
class User:
    def __init__(self, db_name='portfolio.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.current_user_id = None # Stored Logged-in user ID

        self.conn.commit()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def is_strong_password(self, password):
        if (len(password) >= 8 and
                re.search(r"[A-Z]", password) and
                re.search(r"[a-z]", password) and
                re.search(r"[0-9]", password) and
                re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)):
            return True
        return False

    def register_user(self, username, password, security_question, security_answer):
        hashed_password = self.hash_password(password)
        hashed_answer = self.hash_password(security_answer)

        self.cursor.execute(
            "INSERT INTO users (username, password, security_question, security_answer) VALUES (?, ?, ?, ?)",
            (username, hashed_password, security_question, hashed_answer))
        self.conn.commit()

    def login_user(self, username, password):
        hashed_password = self.hash_password(password)
        self.cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
        user = self.cursor.fetchone()

        if user:
            self.current_user_id = username  # Store logged-in username
            return True
        return False

    def get_user_by_username(self, username):
        self.cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return self.cursor.fetchone()

    def reset_password(self, username, new_password):
        hashed_new_password = self.hash_password(new_password)
        self.cursor.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_new_password, username))
        self.conn.commit()

    def fetch_risk_tolerance(self, username):
        self.cursor.execute("SELECT risk_tolerance FROM users WHERE username = ?", (username,))
        result = self.cursor.fetchone()
        if not result or result[0] is None:
            default_tolerance = 'Medium'
            self.cursor.execute('''
                    UPDATE users SET risk_tolerance = ? WHERE username = ?
                ''', (default_tolerance, username))
            self.conn.commit()
            return default_tolerance

        return result[0]  # Return the existing risk tolerance

    def update_risk_tolerance(self, username, new_risk_tolerance):
        self.cursor.execute("UPDATE users SET risk_tolerance = ? WHERE username = ?", (new_risk_tolerance, username))
        self.conn.commit()

#==============
# Register
#==============
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        security_question = request.form['security_question']
        security_answer = request.form['security_answer']
        username_error= None
        password_error = None
        user = User()
        if user.get_user_by_username(username):
            username_error="Username already exists. Please try registering with another username."
            if not user.is_strong_password(password):
                password_error = "Weak password. Please follow the guidelines."
        else:
            if user.is_strong_password(password):
                user.register_user(username, password, security_question, security_answer)
                return render_template('login.html')  # Redirect to login after registration
            else:
                password_error = "Weak password. Please follow the guidelines."
        return render_template('register.html', username_error=username_error, password_error=password_error)
    return render_template('register.html')
    
#==============
# Login
#==============
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User()
        session['login_username'] = username
        if request.form.get('forget_password') == 'true':
            return redirect(url_for('forget_password'))
        elif user.login_user(username, password):
            return redirect(url_for('dashboard', username=username))  # Redirect to page after successful login
        else:
            return render_template('login.html', error="Invalid credentials, please try again.", )
    return render_template('login.html')

@app.route('/forget_password', methods=['GET', 'POST'])
def forget_password():
    username_error = None
    if request.method == 'POST':
        username = request.form['username']
        user = User()
        if user.get_user_by_username(username):
            session['forget_username'] = username  # ✅ Store username in session
            return redirect(url_for('test'))
        else:
            username_error = "Username does not exist."
    return render_template('forget_password.html', username_error=username_error)

@app.route('/test', methods=['GET', 'POST'])
def test():
    security_ans_error = None
    security_question = None
    user = User()
    username = session.get('forget_username')

    if not username:
        return redirect(url_for('forget_password'))  # If no username, go back

        # Get the security question for the username
    user.cursor.execute("SELECT security_question FROM users WHERE username = ?", (username,))
    security = user.cursor.fetchone()
    if security:
        security_question = security[0]
        if request.method == 'POST':
            security_answer = request.form['security_answer']
            hashed_answer = user.hash_password(security_answer)

            user.cursor.execute("SELECT * FROM users WHERE username = ? AND security_answer = ?",
                                (username, hashed_answer))
            if user.cursor.fetchone():
                return render_template('reset_password.html', security_question=security_question)

            else:
                security_ans_error = "Incorrect answer to the security question. Please try again."

    return render_template('test.html', security_question=security_question, security_ans_error=security_ans_error)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    password_error = None
    user = User()
    username = session.get('forget_username')
    if not username:
        return redirect(url_for('forget_password'))

    if request.method == 'POST':
        new_password = request.form['new_password']
        print(new_password)
        if user.is_strong_password(new_password):
            user.reset_password(username, new_password)
            session.pop('forget_username', None)
            print("yellow")
            return redirect(url_for('login'))
        else:
            password_error = "Weak password. Please follow the guidelines."
    return render_template('reset_password.html', password_error=password_error)

#Dashboard ---------------------------------------------------------------------------------
@app.route("/dashboard/<username>")
def dashboard(username):
    conn = init_db()
    portfolio_data = view_portfolio(username, conn)  # Fetch portfolio data for table

    if not portfolio_data:
        portfolio = []
        total_realized_pnl = 0
        total_unrealized_pnl = 0
        total_portfolio_value = 0
    else:
        portfolio = portfolio_data.get("portfolio", [])
        total_realized_pnl = portfolio_data.get("total_realized_pnl",0)
        total_unrealized_pnl = portfolio_data.get("total_unrealized_pnl", 0)
        total_portfolio_value = portfolio_data.get("total_portfolio_value", 0)

    chart_path = visualise_portfolio(username)  # Fetch pie chart data

    is_empty = len(portfolio) == 0 #Check if portfolio is empty

    return render_template(
        "dashboard.html",
        username=username,
        portfolio=portfolio,
        total_realized_pnl=total_realized_pnl,
        total_unrealized_pnl=total_unrealized_pnl,
        total_portfolio_value=total_portfolio_value,
        is_empty=is_empty,
        import_error=None,
        import_success=None
    )

@app.route("/portfolio_chart/<username>")
def portfolio_chart(username):
    chart_path = visualise_portfolio(username)
    if chart_path and os.path.exists(chart_path):
        return send_file(chart_path, mimetype='image/png')
    return "No chart available", 404


@app.route("/export_portfolio/<username>")
def export_portfolio(username):
    conn = init_db()
    portfolio = view_portfolio(username, conn)["portfolio"]  # Fetch portfolio data from DB
    filename = f"portfolio_{username}_{datetime.date.today()}.csv"
    filepath = os.path.join("exports", filename)

    if not os.path.exists("exports"):
        os.makedirs("exports")

    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Ticker", "Shares", "Sector", "Purchase Price", "Live Price", "Unrealized P/L", "Realized P/L"])

        # Iterate through the list of dictionaries
        for stock in portfolio:
            writer.writerow([
                stock["ticker"],
                stock["shares"],
                stock["sector"],
                stock["purchase_price"],
                stock["current_price"],
                stock["unrealized_pnl"],
                stock["realized_pnl"]
            ])

    return send_file(filepath, as_attachment=True)


@app.route("/dashboard/<username>/import", methods=["POST"])
def import_portfolio(username):
    if 'csv_file' not in request.files:
        flash("No file part", "error")
        return render_template('dashboard.html', username=username, import_error="No file selected.")

    file = request.files['csv_file']
    if file.filename == '':
        flash("No selected file", "error")
        return render_template('dashboard.html', username=username, import_error="No selected file.")

    if file:
        filename = secure_filename(f"portfolio_{username}.csv")
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process CSV File
        try:
            with open(file_path, mode="r") as f:
                reader = csv.reader(f)
                headers = next(reader, None)  # Read the header

                expected_headers = ["Ticker", "Shares", "Sector", "Purchase Price", "Live Price", "Unrealized P/L",
                                    "Realized P/L"]
                if headers != expected_headers:
                    flash("Invalid CSV format! Please use the correct template.", "error")
                    return redirect(url_for("dashboard", username=username))

                conn = sqlite3.connect("portfolio.db")
                cursor = conn.cursor()

                for row in reader:
                    try:
                        ticker, shares, sector, purchase_price, live_price, unrealised_pnl, realized_pnl = row
                        shares = int(shares)
                        purchase_price = float(purchase_price)
                        live_price = float(live_price)
                        unrealised_pnl = float(unrealised_pnl)
                        realized_pnl = float(realized_pnl) if realized_pnl != "Not Available" else None

                        cursor.execute('''
                                INSERT INTO portfolios (username, ticker, shares, purchase_price, purchase_date, sale_price, sale_date, realized_profit_loss)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (username, ticker, shares, purchase_price, "Unknown", None, None, realized_pnl))

                    except ValueError:
                        flash(f"Skipping invalid row: {row}", "error")

                conn.commit()
                conn.close()

            flash("Portfolio imported successfully!", "success")
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "error")

    return redirect(url_for("dashboard", username=username))

# Risk Tolerance -----------------------------------------------------------------
@app.route('/dashboard/<username>/risk_tolerance', methods=['GET', 'POST'])
def risk_tolerance(username):
    user = User()
    username = session.get('login_username')
    new_risk_tolerance = None

    if request.method == 'POST':
        new_risk_tolerance = request.form['risk_tolerance']
        user.update_risk_tolerance(username, new_risk_tolerance)
        print(new_risk_tolerance)
    else:
        new_risk_tolerance = user.fetch_risk_tolerance(username)
    return render_template('risk_tolerance.html', risk_tolerance=new_risk_tolerance, username=username)

# Log out -----------------------------------------------------------------
@app.route("/logout", methods=["POST"])
def logout():
    # Clear the session
    session.pop('username', None)

    # Redirect to the homepage after logout
    return redirect(url_for('homepage'))

cache = {}
cache_expired = timedelta(minutes=30)

# Fetch Stock Info -----------------------------------------------------------------
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")
    returns = hist['Close'].pct_change().dropna()
    return {
        'Ticker': ticker,
        'Company': info.get('shortName', 'N/A'),
        'Sector': info.get('sector', 'N/A'),
        'Industry': info.get('industry', 'N/A'),
        'Market Cap': info.get('marketCap', 'N/A'),
        'Previous Close': info.get('previousClose', 'N/A'),
        'Trailing PE': info.get('trailingPE', 'N/A'),
        'Forward PE': info.get('forwardPE', 'N/A'),
        'Returns': returns.to_dict()  # Optional, if you want to send it
    }

def generate_chart_base64(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="6mo")

    if history.empty:
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(history.index, history['Close'], label=f"{ticker} Price", color="blue")
    plt.title(f"{ticker} Stock Price (Last 6 Months)")
    plt.xlabel("Date")
    plt.ylabel("Closing Price ($)")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_img

def recommend_stocks(ticker, risk_tolerance):
    stock_data = fetch_stock_data(ticker)
    risk_info = calculate_risk_info(stock_data)

    if stock_data['Trailing PE'] is None or risk_info is None:
        return {"error": f"No P/E ratio or risk data available for {ticker}."}

    pe_ratio = stock_data['Trailing PE']
    industry_avg_pe = 20  # Assume industry avg P/E

    if risk_tolerance == "Low":
        sharpe_threshold = 1.5
    elif risk_tolerance == "Medium":
        sharpe_threshold = 1.0
    else:
        sharpe_threshold = 0.5

    if pe_ratio < industry_avg_pe * 0.8 and risk_info["sharpe_ratio"] > sharpe_threshold:
        recommendation = f"✅ {ticker} is undervalued and within your risk tolerance. Consider adding."
    elif pe_ratio > industry_avg_pe * 1.2 or risk_info["sharpe_ratio"] < sharpe_threshold:
        recommendation = f"⚠️ {ticker} may be overvalued or outside your risk tolerance. Consider avoiding."
    else:
        recommendation = f"⏸️ {ticker} is fairly valued and within risk tolerance. Consider holding."

    return {
        "stock_data": stock_data,
        "risk_info": risk_info,
        "recommendation": recommendation
    }

def show_overall_results(ticker, sentiment_scores):
    if not sentiment_scores:
        return {"label": "Neutral", "score": 0}

    scores = [score for _, score, _ in sentiment_scores]
    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.05:
        label = "Positive"
    elif avg_score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"label": label, "score": avg_score}

def format_headline_data(sentiment_scores):
    data = []
    for headline, score, url in sentiment_scores:
        data.append({
            "headline": headline,
            "score": round(score, 2),
            "url": url
        })
    return data

@app.route("/get_stock_info")
def get_stock_info():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Missing ticker symbol"}), 400

    try:
        stock_data = fetch_stock_data(ticker)
        chart_base64 = generate_chart_base64(ticker)
        if chart_base64 is None:
            return jsonify({"error": "Chart could not be generated."}), 500

    # Render the template and pass the data to it
        return render_template("Stock_Information.html", stock_data=stock_data, chart=chart_base64)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_stock_recommendation_and_metrics')
def get_stock_recommendation_and_metrics():
    ticker = request.args.get('ticker', '').upper()

    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    try:
        session['risk_tolerance']=risk_tolerance  
        result = recommend_stocks(ticker, risk_tolerance)

        if "error" in result:
            return jsonify({"error": result["error"]}), 404
    
        # Render the template and pass the data to it
        return render_template(
            "Recommendations.html",
            stock_data=result["stock_data"],
            risk_info=result["risk_info"],
            recommendation=result["recommendation"]
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_market_sentiment")
def market_sentiment():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Missing ticker"}), 400

    try:
        result = fetch_market_sentiment(ticker)
        if not result:
            return jsonify({"error": "Failed to retrieve data"}), 500

        overall_data, headline_data = result

        # Render the template and pass the data to it
        return render_template(
            "Market_Sentiment.html",
            ticker=ticker,
            overall_sentiment=overall_data,
            headlines=headline_data
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
