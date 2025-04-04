from flask import Flask, render_template, url_for, request, redirect, flash, send_file, jsonify, session
import sqlite3
import json
import csv
import datetime
from db import init_db, view_portfolio, visualise_portfolio, load_csv_data, insert_data_to_db, fetch_stock_data
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"
# Define the folder for uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}



conn = init_db()
usa_large_companies, stock_info, nasdaq_data, sp_data = load_csv_data()
insert_data_to_db(conn, stock_info)


@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        return redirect(url_for('homepage'))
    return render_template('homepage.html')


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


@app.route("/logout", methods=["POST"])
def logout():
    # Clear the session
    session.pop('username', None)

    # Redirect to the homepage after logout
    return redirect(url_for('homepage'))

if __name__ == '__main__':
    app.run()
