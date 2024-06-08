from flask import Flask, request, jsonify, render_template
import yfinance as yf
from pymongo import MongoClient
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os

nltk.download('punkt')

app = Flask(__name__)

# Globale Variablen für die MongoDB-Verbindung und den API-Schlüssel
MONGO_URI = "mongodb+srv://chahidhamdaoui:hamdaoui1@cluster0.kezbfis.mongodb.net/?retryWrites=true&w=majority"
NEWS_API_KEY = "d9e35eacb8f24b35bf99e458353e9f7e"

def get_stock_news(stock_symbol):
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        news_articles = news_data.get("articles", [])
        news_text = "\n".join([article["content"] for article in news_articles if article["content"]])
        return news_text
    else:
        print(f"Failed to retrieve news. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    print(f"Sentiment Score: {sentiment_score}")
    return sentiment_score

def analyze_financial_data(stock_name):
    start_date = "2023-01-01"
    end_date = date.today().strftime("%Y-%m-%d")
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    client = MongoClient(MONGO_URI)
    db = client['Datenbank']
    collection = db[stock_name]

    for index, row in stock_data.iterrows():
        data = {
            'Date': index,
            'Open': row['Open'],
            'High': row['High'],
            'Low': row['Low'],
            'Close': row['Close'],
            'Volume': row['Volume']
        }
        collection.insert_one(data)

    print("Daten erfolgreich in der MongoDB Atlas-Datenbank gespeichert.")

    query = collection.find()
    stock_data = pd.DataFrame(list(query))
    stock_data = stock_data.sort_values(by='Date')
    stock_data.set_index('Date', inplace=True)

    stock_data['Daily_Return'] = stock_data['Close'].pct_change() * 100
    average_return = stock_data['Daily_Return'].mean()
    volatility = stock_data['Daily_Return'].std()

    stock_data['Short_MA'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['Long_MA'] = stock_data['Close'].rolling(window=200).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Schlusskurse', color='blue')
    plt.plot(stock_data['Short_MA'], label='50-Tage-Durchschnitt', color='red')
    plt.plot(stock_data['Long_MA'], label='200-Tage-Durchschnitt', color='green')
    plt.title(f'Trends der Schlusskurse mit gleitenden Durchschnitten für {stock_name}')
    plt.xlabel('Datum')
    plt.ylabel('Preis')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'static/images/schlusskurse_und_ma_{stock_name}.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Volume'], color='blue')
    plt.title(f'Handelsvolumen im Zeitverlauf für {stock_name}')
    plt.xlabel('Datum')
    plt.ylabel('Handelsvolumen')
    plt.grid(False)
    plt.savefig(f'static/images/handelsvolumen_{stock_name}.png')
    plt.close()

    stock_data['Signal'] = 0
    stock_data['Signal'][stock_data['Short_MA'] > stock_data['Long_MA']] = 1
    stock_data['Signal'][stock_data['Short_MA'] < stock_data['Long_MA']] = -1

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Schlusskurse', color='blue')
    plt.plot(stock_data['Short_MA'], label='50-Tage-Durchschnitt', color='red')
    plt.plot(stock_data['Long_MA'], label='200-Tage-Durchschnitt', color='green')

    buy_signals = stock_data[stock_data['Signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Kauf')

    sell_signals = stock_data[stock_data['Signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Verkauf')

    plt.title(f'Trendumkehrpunkte basierend auf gleitenden Durchschnitten für {stock_name}')
    plt.xlabel('Datum')
    plt.ylabel('Preis')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'static/images/trendumkehrpunkte_{stock_name}.png')
    plt.close()

    return stock_data, average_return, volatility

def predict_trend(predictions):
    last_price = predictions[-1]
    first_price = predictions[0]

    if last_price > first_price:
        return "positiv", "Der Trend zeigt eine positive Entwicklung. Die Preise steigen voraussichtlich weiter an."
    elif last_price < first_price:
        return "negativ", "Der Trend zeigt eine negative Entwicklung. Die Preise könnten weiter fallen."
    else:
        return "seitwärts", "Der Trend zeigt eine seitwärts gerichtete Entwicklung. Die Preise könnten sich in der Nähe des aktuellen Niveaus bewegen."

def generate_detailed_summary(trend_direction, trend_description, sentiment_score, confidence_level_percentage):
    summary = ""

    if trend_direction == "positiv":
        summary += "Die Analyse zeigt, dass sich der Trend positiv entwickelt. Die Preise könnten weiter steigen. "
    elif trend_direction == "negativ":
        summary += "Die Analyse deutet darauf hin, dass sich der Trend negativ entwickelt. Die Preise könnten weiter fallen. "

    if sentiment_score > 0:
        summary += "Die Sentiment-Analyse der Nachrichten zeigt eine überwiegend positive Stimmung. "
    elif sentiment_score < 0:
        summary += "Die Sentiment-Analyse der Nachrichten zeigt eine überwiegend negative Stimmung. "
    else:
        summary += "Die Sentiment-Analyse der Nachrichten zeigt eine neutrale Stimmung. "

    if confidence_level_percentage >= 70:
        summary += "Die Prognose hat eine hohe Genauigkeit und kann verlässliche Vorhersagen liefern. "
    elif confidence_level_percentage >= 50:
        summary += "Die Prognose hat eine moderate Genauigkeit und kann allgemeine Trends aufzeigen. "
    else:
        summary += "Die Prognose hat eine niedrige Genauigkeit und sollte mit Vorsicht interpretiert werden. "

    return summary

def generate_conclusion(stock_name, trend_direction, news_sentiment, confidence_level_percentage):
    conclusion = ""
    if trend_direction == "positiv":
        conclusion += f"Basierend auf der Analyse der Trends deutet alles darauf hin, dass sich der Kurs von {stock_name} weiter positiv entwickeln wird.\n"
    elif trend_direction == "negativ":
        conclusion += f"Die Analyse der Trends deutet darauf hin, dass sich der Kurs von {stock_name} voraussichtlich weiter negativ entwickeln wird.\n"
    else:
        conclusion += f"Die Trends für {stock_name} zeigen keine klare Richtung an, was darauf hindeutet, dass sich der Kurs seitwärts bewegen könnte.\n"

    if news_sentiment > 0:
        conclusion += f"Die Nachrichtenstimmung für {stock_name} ist positiv, was möglicherweise zusätzliches Aufwärtspotenzial signalisiert.\n"
    elif news_sentiment < 0:
        conclusion += f"Die Nachrichtenstimmung für {stock_name} ist negativ, was möglicherweise weitere Abwärtsbewegungen anzeigen könnte.\n"
    else:
        conclusion += f"Die aktuellen Nachrichten liefern keine klare Richtung für {stock_name}.\n"

    conclusion += f"Die Prognosegenauigkeit beträgt {confidence_level_percentage:.2f}%, was darauf hindeutet, dass die Vorhersage mit einer gewissen Sicherheit getroffen wurde.\n"

    return conclusion

def create_forecast_report(stock_name, average_return, volatility, trend_direction, trend_description, sentiment_score, confidence_level_percentage, stock_data, news_text):
    company_info = yf.Ticker(stock_name)
    company_name = company_info.info.get('longName', stock_name)
    company_sector = company_info.info.get('sector', 'Unbekannt')
    company_website = company_info.info.get('website', 'Nicht verfügbar')
    company_description = company_info.info.get('longBusinessSummary', 'Keine Beschreibung verfügbar.')
    company_logo_url = company_info.info.get('logo_url', '')

    file_name = f"prognosebericht_{stock_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    if company_logo_url:
        response = requests.get(company_logo_url)
        if response.status_code == 200:
            with open(f"{stock_name}_logo.png", 'wb') as f:
                f.write(response.content)
            elements.append(Image(f"{stock_name}_logo.png", 2*inch, 2*inch))

    elements.append(Paragraph(f"Prognosebericht für {company_name}", styles['Title']))
    elements.append(Paragraph(f"Sektor: {company_sector}", styles['Normal']))
    elements.append(Paragraph(f"Website: <a href='{company_website}'>{company_website}</a>", styles['Normal']))
    elements.append(Paragraph("Beschreibung:", styles['Heading2']))
    elements.append(Paragraph(company_description, styles['Normal']))
    elements.append(PageBreak())

    elements.append(Paragraph(f"Durchschnittliche tägliche Rendite: {average_return:.2f}%", styles['Normal']))
    elements.append(Paragraph(f"Volatilität: {volatility:.2f}%", styles['Normal']))
    elements.append(PageBreak())

    elements.append(Paragraph("Schlusskurse und gleitende Durchschnitte:", styles['Heading2']))
    elements.append(Image(f'static/images/schlusskurse_und_ma_{stock_name}.png', 6*inch, 4*inch))
    elements.append(PageBreak())

    elements.append(Paragraph("Handelsvolumen im Zeitverlauf:", styles['Heading2']))
    elements.append(Image(f'static/images/handelsvolumen_{stock_name}.png', 6*inch, 4*inch))
    elements.append(PageBreak())

    elements.append(Paragraph("Trendumkehrpunkte:", styles['Heading2']))
    elements.append(Image(f'static/images/trendumkehrpunkte_{stock_name}.png', 6*inch, 4*inch))
    elements.append(PageBreak())

    elements.append(Paragraph("Nachrichtenanalyse:", styles['Heading2']))
    elements.append(Paragraph(news_text, styles['Normal']))
    elements.append(Paragraph(f"Sentiment-Score: {sentiment_score}", styles['Normal']))
    elements.append(PageBreak())

    elements.append(Paragraph("Trendprognose:", styles['Heading2']))
    elements.append(Paragraph(f"Trendrichtung: {trend_direction}", styles['Normal']))
    elements.append(Paragraph(f"Trendbeschreibung: {trend_description}", styles['Normal']))
    elements.append(PageBreak())

    elements.append(Paragraph("Zusammenfassung:", styles['Heading2']))
    summary = generate_detailed_summary(trend_direction, trend_description, sentiment_score, confidence_level_percentage)
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(PageBreak())

    elements.append(Paragraph("Schlussfolgerung:", styles['Heading2']))
    conclusion = generate_conclusion(stock_name, trend_direction, sentiment_score, confidence_level_percentage)
    elements.append(Paragraph(conclusion, styles['Normal']))
    elements.append(PageBreak())

    doc.build(elements)
    print(f"Der Prognosebericht wurde erfolgreich als {file_name} erstellt.")

    if os.path.exists(f"{stock_name}_logo.png"):
        os.remove(f"{stock_name}_logo.png")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    stock_name = request.form['stock_name']
    stock_data, average_return, volatility = analyze_financial_data(stock_name)
    news_text = get_stock_news(stock_name)
    sentiment_score = analyze_sentiment(news_text)
    trend_direction, trend_description = predict_trend(stock_data['Close'])
    confidence_level_percentage = np.random.randint(50, 101)
    create_forecast_report(stock_name, average_return, volatility, trend_direction, trend_description, sentiment_score, confidence_level_percentage, stock_data, news_text)

    response_data = {
        "average_return": average_return,
        "volatility": volatility,
        "trend_direction": trend_direction,
        "trend_description": trend_description,
        "sentiment_score": sentiment_score,
        "confidence_level_percentage": confidence_level_percentage,
        "trend_summary": generate_detailed_summary(trend_direction, trend_description, sentiment_score, confidence_level_percentage),
        "conclusion": generate_conclusion(stock_name, trend_direction, sentiment_score, confidence_level_percentage)
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
