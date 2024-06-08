import yfinance as yf
from pymongo import MongoClient
from datetime import date, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import nltk
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os

nltk.download('punkt')

# Funktion zum Abrufen von Nachrichten und Durchführen der Sentimentanalyse
import requests
from textblob import TextBlob

# Funktion zur Abfrage von Nachrichten
def get_stock_news(stock_symbol):
    api_key = "d9e35eacb8f24b35bf99e458353e9f7e"
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={api_key}"
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

# Funktion zur Sentiment-Analyse mit TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    # Markierung des sentiment_score
    print(f"Sentiment Score: {sentiment_score}")
    return sentiment_score

# Funktion zur Erstellung einer einfachen Prognose
def make_forecast(sentiment_score):
    # Beispielhafte Prognose: Wenn der Sentiment-Score positiv ist, sagen wir "Kaufen", sonst "Verkaufen"
    if sentiment_score > 0:
        return "Buy"
    else:
        return "Sell"

# Funktion zur Analyse der Finanzdaten
def analyze_financial_data(stock_symbol):
    news_text = get_stock_news(stock_symbol)
    if news_text:
        sentiment_score = analyze_sentiment(news_text)
        forecast = make_forecast(sentiment_score)
        print(f"Forecast for {stock_symbol}: {forecast}")
    else:
        print("No news text available for sentiment analysis.")

# Funktion zur Analyse der Finanzdaten


def analyze_financial_data(stock_name):
    start_date = "2023-01-01"
    end_date = date.today().strftime("%Y-%m-%d")
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    client = MongoClient("mongodb+srv://chahidhamdaoui:hamdaoui1@cluster0.kezbfis.mongodb.net/?retryWrites=true&w=majority")
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
    #stock_data['Date'] = pd.to_datetime(stock_data['Date'])
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
    plt.savefig(f'schlusskurse_und_ma_{stock_name}.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Volume'], color='blue')
    plt.title(f'Handelsvolumen im Zeitverlauf für {stock_name}')
    plt.xlabel('Datum')
    plt.ylabel('Handelsvolumen')
    plt.grid(False)
    plt.savefig(f'handelsvolumen_{stock_name}.png')
    plt.show()

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
    plt.savefig(f'trendumkehrpunkte_{stock_name}.png')
    plt.show()

    return stock_data, average_return, volatility

# Beispielaufruf
# Funktion zur Trendvorhersage
def predict_trend(predictions):
    last_price = predictions[-1]
    first_price = predictions[0]

    if last_price > first_price:
        return "positiv", "Der Trend zeigt eine positive Entwicklung. Die Preise steigen voraussichtlich weiter an."
    elif last_price < first_price:
        return "negativ", "Der Trend zeigt eine negative Entwicklung. Die Preise könnten weiter fallen."
    else:
        return "seitwärts", "Der Trend zeigt eine seitwärts gerichtete Entwicklung. Die Preise könnten sich in der Nähe des aktuellen Niveaus bewegen."


# Funktion zum Generieren eines ausführlichen Fazits basierend auf den Analyseergebnissen
def generate_detailed_summary(trend_direction, trend_description, sentiment_score, confidence_level_percentage):
    summary = ""

    # Fazit zum Trend
    if trend_direction == "positiv":
        summary += "Die Analyse zeigt, dass sich der Trend positiv entwickelt. Die Preise könnten weiter steigen. "
    elif trend_direction == "negativ":
        summary += "Die Analyse deutet darauf hin, dass sich der Trend negativ entwickelt. Die Preise könnten weiter fallen. "

    # Fazit zur Sentiment-Analyse
    if sentiment_score > 0:
        summary += "Die Sentiment-Analyse der Nachrichten zeigt eine überwiegend positive Stimmung. "
    elif sentiment_score < 0:
        summary += "Die Sentiment-Analyse der Nachrichten zeigt eine überwiegend negative Stimmung. "
    else:
        summary += "Die Sentiment-Analyse der Nachrichten zeigt eine neutrale Stimmung. "

    # Fazit zur Prognosegenauigkeit
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




# Funktion zur Erstellung eines Prognoseberichts
def create_forecast_report(stock_name, average_return, volatility, trend_direction, trend_description, sentiment_score, confidence_level_percentage, stock_data, news_text):
    company_info = yf.Ticker(stock_name)
    company_name = company_info.info.get('longName', stock_name)
    company_sector = company_info.info.get('sector', 'Unbekannt')
    company_website = company_info.info.get('website', 'Nicht verfügbar')
    company_description = company_info.info.get('longBusinessSummary', 'Keine Beschreibung verfügbar.')
    company_logo_url = company_info.info.get('logo_url', '')

    file_name = f"prognosebericht_{stock_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []

    header_text = f"<br/><br/><font name='Helvetica-Bold' size='20'>Prognosebericht für {company_name}</font>"
    header_text += f"<br/><font name='Helvetica' size='12'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</font>"
    header_text += "<br/><font name='Helvetica' size='18'>UnChained International</font><br/><br/>"
    content.append(Paragraph(header_text, styles['Title']))

    if company_logo_url:
        logo_path = f'logo_{stock_name}.png'
        with open(logo_path, 'wb') as f:
            f.write(requests.get(company_logo_url).content)
        content.append(Image(logo_path, 2*inch, 2*inch))

    company_info_text = "<font name='Helvetica-Bold'>Unternehmensinformationen:</font><br/>"
    company_info_text += f"<font name='Helvetica'>Name: {company_name}<br/>"
    company_info_text += f"Branche: {company_sector}<br/>"
    company_info_text += f"Website: <a href='{company_website}' color='blue'>{company_website}</a><br/><br/></font>"
    company_info_text += f"<font name='Helvetica'>Beschreibung: {company_description}<br/><br/></font>"
    content.append(Paragraph(company_info_text, styles['Normal']))

    financial_info_text = "<font name='Helvetica-Bold'>Finanzinformationen:</font><br/>"
    financial_info_text += f"<font name='Helvetica'>Durchschnittliche tägliche Rendite: {average_return:.2f}%<br/>"
    financial_info_text += f"Volatilität der täglichen Renditen: {volatility:.2f}%<br/><br/></font>"
    content.append(Paragraph(financial_info_text, styles['Normal']))

    trend_info_text = "<font name='Helvetica-Bold'>Trendvorhersage:</font><br/>"
    trend_info_text += f"<font name='Helvetica'>Der Trend zeigt eine {trend_direction} Entwicklung. Die Preise {trend_description}<br/><br/></font>"
    content.append(Paragraph(trend_info_text, styles['Normal']))

    sentiment_info_text = "<font name='Helvetica-Bold'>Nachrichtenstimmungswert:</font><br/>"
    sentiment_info_text += f"<font name='Helvetica'>{sentiment_score:.4f}<br/><br/></font>"
    content.append(Paragraph(sentiment_info_text, styles['Normal']))

    confidence_info_text = "<font name='Helvetica-Bold'>Prozentuale Sicherheit der Prognose:</font><br/>"
    confidence_info_text += f"<font name='Helvetica'>{confidence_level_percentage:.2f}%<br/><br/></font>"
    content.append(Paragraph(confidence_info_text, styles['Normal']))

    content.append(PageBreak())
    content.append(Paragraph("<font name='Helvetica-Bold'>Grafiken:</font><br/><br/>", styles['Heading2']))

    img_path = f'schlusskurse_und_ma_{stock_name}.png'
    content.append(Image(img_path, 6*inch, 4*inch))
    content.append(Paragraph("<font name='Helvetica'>Dieser Graph zeigt die Schlusskurse sowie die 50-Tage- und 200-Tage-Durchschnitte, um die Trends der Aktienpreise zu veranschaulichen.</font><br/><br/>", styles['Normal']))
     # Analyse für den ersten Graphen (Schlusskurse und Durchschnitte)
    analysis_text1 = "<font name='Helvetica-Bold'>Analyse der Schlusskurse und Durchschnitte:</font><br/>"
    analysis_text1 += "<font name='Helvetica'>Die Grafik zeigt die Schlusskurse der Aktie sowie die 50-Tage- und 200-Tage-Durchschnitte. Ein Anstieg der Schlusskurse über die Durchschnitte deutet auf einen möglichen Aufwärtstrend hin, während ein Rückgang unter die Durchschnitte auf einen möglichen Abwärtstrend hindeuten könnte. Ein weiterer Aspekt ist die Divergenz zwischen den beiden Durchschnitten - eine Konvergenz könnte auf eine Trendumkehr hinweisen, während eine Divergenz auf einen bestehenden Trend hindeuten könnte.<br/><br/></font>"
    content.append(Paragraph(analysis_text1, styles['Normal']))

    content.append(PageBreak())

    img_path = f'handelsvolumen_{stock_name}.png'
    content.append(Image(img_path, 6*inch, 4*inch))
    content.append(Paragraph("<font name='Helvetica'>Dieser Graph zeigt das Handelsvolumen im Zeitverlauf, was auf die Handelsaktivität und das Interesse an der Aktie hinweist.</font><br/><br/>", styles['Normal']))
    analysis_text2 = "<font name='Helvetica-Bold'>Analyse des Handelsvolumens:</font><br/>"
    analysis_text2 += "<font name='Helvetica'>Das Handelsvolumen gibt Aufschluss über die Aktivität und das Interesse der Marktteilnehmer an der Aktie. Ein Anstieg des Handelsvolumens kann auf eine verstärkte Marktbeteiligung und mögliche Kursbewegungen hinweisen, während ein Rückgang des Volumens auf eine geringere Aktivität und mögliche Stabilität oder Konsolidierung hindeuten könnte. Die Analyse des Handelsvolumens in Verbindung mit Preisbewegungen kann wichtige Einblicke in die Stärke und Richtung des Trends liefern.<br/><br/></font>"
    content.append(Paragraph(analysis_text2, styles['Normal']))

    content.append(PageBreak())

    img_path = f'trendumkehrpunkte_{stock_name}.png'
    content.append(Image(img_path, 6*inch, 4*inch))
    content.append(Paragraph("<font name='Helvetica'>Dieser Graph zeigt die Trendumkehrpunkte basierend auf den gleitenden Durchschnitten, um potenzielle Kauf- und Verkaufszeitpunkte zu identifizieren.</font><br/><br/>", styles['Normal']))
    analysis_text3 = "<font name='Helvetica-Bold'>Analyse der Trendumkehrpunkte:</font><br/>"
    analysis_text3 += "<font name='Helvetica'>Die Trendumkehrpunkte basierend auf den gleitenden Durchschnitten können potenzielle Wendepunkte im Preisverlauf anzeigen. Ein Wechsel von einem Aufwärts- zu einem Abwärtstrend oder umgekehrt kann wichtige Handelssignale liefern. Die Analyse der Trendumkehrpunkte in Verbindung mit anderen technischen Indikatoren kann dabei helfen, potenzielle Kauf- und Verkaufszeitpunkte zu identifizieren und Risiken zu managen.<br/><br/></font>"
    content.append(Paragraph(analysis_text3, styles['Normal']))

    content.append(PageBreak())

    img_path = f'tats_vs_vorhersage_{stock_name}.png'
    content.append(Image(img_path, 6*inch, 4*inch))
    content.append(Paragraph("<font name='Helvetica'>Dieser Graph zeigt die tatsächlichen und vorhergesagten Preise, um die Genauigkeit des Modells bei der Vorhersage der Aktienpreise zu veranschaulichen.</font><br/><br/>", styles['Normal']))
    analysis_text4 = "<font name='Helvetica-Bold'>Analyse der tatsächlichen vs. vorhergesagten Preise:</font><br/>"
    analysis_text4 += "<font name='Helvetica'>Der Vergleich zwischen den tatsächlichen und vorhergesagten Preisen ermöglicht es, die Genauigkeit des Vorhersagemodells zu bewerten. Eine Annäherung der vorhergesagten Preise an die tatsächlichen Preise deutet auf eine gute Vorhersageleistung hin, während größere Abweichungen auf Verbesserungsbedarf hinweisen können. Es ist wichtig, die Genauigkeit der Vorhersagen im Kontext der Marktbedingungen und anderer Einflussfaktoren zu bewerten, um fundierte Handelsentscheidungen zu treffen.<br/><br/></font>"
    content.append(Paragraph(analysis_text4, styles['Normal']))

    content.append(PageBreak())
    content.append(Paragraph("<font name='Helvetica-Bold'>Fazit:</font><br/><br/>", styles['Heading2']))

    conclusion_text = "<font name='Helvetica-Bold'></font><br/><br/>"
    conclusion = generate_detailed_summary(stock_name, trend_direction, sentiment_score, confidence_level_percentage)
    conclusion_text += f"<font name='Helvetica'>{conclusion}<br/><br/></font>"
    content.append(Paragraph(conclusion_text, styles['Normal']))


    content.append(PageBreak())
    content.append(Paragraph("<font name='Helvetica-Bold'>NEWS:</font><br/><br/>", styles['Heading2']))



    # Nachrichten als Anhang hinzufügen
    if news_text:
        news_text_cleaned = "".join(c for c in news_text if ord(c) < 128)
        news_attachment_text = "<font name='Helvetica-Bold'>Nachrichten:</font><br/>"
        news_attachment_text += f"<font name='Helvetica'>{news_text_cleaned}<br/><br/></font>"
        content.append(Paragraph(news_attachment_text, styles['Normal']))

    doc.build(content)
    print(f"Prognosebericht für {stock_name} wurde erfolgreich erstellt: {file_name}")


def main():
    stocks = ["AAPL", "MSFT", "NVDA", "^GDAXI"]
    additional_stock = input("Möchten Sie eine weitere Aktie analysieren? Geben Sie den Namen der Aktie ein oder 'NO' zum Abbrechen: ")
    if additional_stock.upper() != "NO":
        stocks.append(additional_stock)


    for stock_name in stocks:
        stock_data, average_return, volatility = analyze_financial_data(stock_name)



        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

        train_size = int(len(normalized_data) * 0.8)
        train_data = normalized_data[:train_size]
        test_data = normalized_data[train_size:]

        n = 30
        X_train = []
        y_train = []
        for i in range(n, len(train_data)):
            X_train.append(train_data[i-n:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        print(f"Training des RNN-Modells für {stock_name}...")
        model.fit(X_train, y_train, epochs=100, batch_size=32)
        print(f"RNN-Modell für {stock_name} erfolgreich trainiert.")

        train_predictions = model.predict(X_train)

        train_predictions = scaler.inverse_transform(train_predictions)

        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index[:train_size], scaler.inverse_transform(train_data), color='blue', label='Tatsächliche Preise')
        plt.plot(stock_data.index[n:train_size], train_predictions, color='red', label='Vorhergesagte Preise')
        plt.title(f'Tatsächliche vs. vorhergesagte Preise für {stock_name}')
        plt.xlabel('Datum')
        plt.ylabel('Preis')
        plt.legend()
        plt.savefig(f'tats_vs_vorhersage_{stock_name}.png')
        plt.show()

        trend_direction, trend_description = predict_trend(train_predictions)

        confidence_level_scalar = np.mean(np.abs(train_predictions - scaler.inverse_transform(train_data)[n:train_size]))
        confidence_level_percentage = (1 - confidence_level_scalar / np.mean(scaler.inverse_transform(train_data)[n:train_size])) * 100




    for stock_name in stocks:
        stock_data, average_return, volatility = analyze_financial_data(stock_name)
        news_text = get_stock_news(stock_name)
        sentiment_score = analyze_sentiment(news_text)
        create_forecast_report(stock_name, average_return, volatility, trend_direction, trend_description, sentiment_score, confidence_level_percentage, stock_data, news_text)
        detailed_summary = generate_detailed_summary(trend_direction, trend_description, sentiment_score, confidence_level_percentage)
        print("Ausführliches Fazit:")
        print(detailed_summary)


def get_current_price(stock_name):
    try:
        stock = yf.Ticker(stock_name)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        return current_price
    except Exception as e:
        print(f"Fehler beim Abrufen des aktuellen Kurses für {stock_name}: {e}")
        return None

# Liste der Aktien
stocks = ["AAPL", "MSFT", "NVDA", "^GDAXI"]

# Aktuellen Kurs für jede Aktie anzeigen
for stock_name in stocks:
    current_price = get_current_price(stock_name)
    if current_price is not None:
        print(f"Aktueller Kurs für {stock_name}: {current_price}")

if __name__ == "__main__":
    main()
