import pandas as pd
import streamlit as st
import requests
import io
from datetime import datetime, timedelta
import plotly.express as px
import logging
import sqlite3
import yfinance as yf
from typing import Optional, List
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# Theme mapping for tabs
theme_mapping = {
    "Indexes": [
        "SPY", "QQQ", "IWM", "DIA", "SMH"
    ],
    "Bull Leverage ETF": [
        "SPXL", "UPRO", "TQQQ", "SOXL", "UDOW", "FAS", "SPUU", "TNA"
    ],
    "Bear Leverage ETF": [
        "SQQQ", "SPXS", "SOXS", "SDOW", "FAZ","SPDN", "TZA", "SPXU"
    ],
    "Volatility": [
        "VXX", "VIXY", "UVXY"
    ],
    "Bonds": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "AGG"
    ],
    "Commodities": [
        "SPY", "GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"
    ],
    "Nuclear Power": [
        "CEG", "NNE", "GEV", "OKLO", "UUUU", "ASPI", "CCJ"
    ],
    "Crypto": [
        "IBIT", "FBTC", "MSTR", "COIN", "HOOD", "ETHU"
    ],
    "Metals": [
        "GLD", "SLV", "GDX", "GDXJ", "IAU", "SIVR"
    ],
    "Real Estate": [
        "VNQ", "IYR", "XHB", "XLF", "SPG", "PLD", "AMT", "DLR"
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "NKE", "MCD", "DIS", "LOW", "TGT", "LULU"
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "CL", "KMB", "MDLZ", "GIS"
    ],
    "Utilities": [
        "XLU", "DUK", "SO", "D", "NEE", "EXC", "AEP", "SRE", "ED"
    ],
    "Telecommunications": [
        "XLC", "T", "VZ", "TMUS", "S", "LUMN", "VOD"
    ],
    "Materials": [
        "XLB", "XME", "XLI", "FCX", "NUE", "DD", "APD", "LIN", "IFF"
    ],
    "Transportation": [
        "UPS", "FDX", "DAL", "UAL", "LUV", "CSX", "NSC", "KSU", "WAB"
    ],
    "Aerospace & Defense": [
        "LMT", "BA", "NOC", "RTX", "GD", "HII", "LHX", "COL", "TXT"
    ],
    "Retail": [
        "AMZN", "WMT", "TGT", "COST", "HD", "LOW", "TJX", "M", "KSS"
    ],
    "Automotive": [
        "TSLA", "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "BYDDF", "FCAU"
    ],
    "Pharmaceuticals": [
        "PFE", "MRK", "JNJ", "ABBV", "BMY", "GILD", "AMGN", "LLY", "VRTX"
    ],
    "Biotechnology": [
        "AMGN", "REGN", "ILMN", "VRTX", "CRSP", "MRNA", "BMRN", "ALNY",
        "SRPT", "EDIT", "NTLA", "BEAM", "BLUE", "FATE", "SANA"
    ],
    "Insurance": [
        "AIG", "PRU", "MET", "UNM", "LNC", "TRV", "CINF", "PGR", "ALL"
    ],
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
        "ORCL", "CRM", "ADBE", "INTC", "CSCO", "QCOM", "TXN", "IBM",
        "NOW", "AVGO", "INTU", "PANW", "SNOW"
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW",
        "COF", "MET", "AIG", "BK", "BLK", "TFC", "USB", "PNC", "CME", "SPGI"
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV", "TMO", "AMGN", "GILD",
        "CVS", "MDT", "BMY", "ABT", "DHR", "ISRG", "SYK", "REGN", "VRTX",
        "CI", "ZTS"
    ],
    "Consumer": [
        "WMT", "PG", "KO", "PEP", "COST", "MCD", "DIS", "NKE", "SBUX",
        "LOW", "TGT", "HD", "CL", "MO", "KHC", "PM", "TJX", "DG", "DLTR", "YUM"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "VLO",
        "XLE", "HES", "WMB", "KMI", "OKE", "HAL", "BKR", "FANG", "DVN",
        "TRGP", "APA"
    ],
    "Industrials": [
        "CAT", "DE", "UPS", "FDX", "BA", "HON", "UNP", "MMM", "GE", "LMT",
        "RTX", "GD", "CSX", "NSC", "WM", "ETN", "ITW", "EMR", "PH", "ROK"
    ],
    "Semiconductors": [
        "NVDA", "AMD", "QCOM", "TXN", "INTC", "AVGO", "ASML", "KLAC",
        "LRCX", "AMAT", "ADI", "MCHP", "ON", "STM", "MPWR", "TER", "ENTG",
        "SWKS", "QRVO", "LSCC"
    ],
    "Cybersecurity": [
        "CRWD", "PANW", "ZS", "FTNT", "S", "OKTA", "CYBR", "RPD", "NET",
        "QLYS", "TENB", "VRNS", "SPLK", "CHKP", "FEYE", "DDOG", "ESTC",
        "FSLY", "MIME", "KNBE"
    ],
    "Quantum Computing": [
        "IBM", "GOOGL", "MSFT", "RGTI", "IONQ", "QUBT", "HON", "QCOM",
        "INTC", "AMAT", "MKSI", "NTNX", "XERI", "QTUM", "FORM",
        "LMT", "BA", "NOC", "ACN"
    ],
    "Clean Energy": [
        "TSLA", "ENPH", "FSLR", "NEE", "PLUG", "SEDG", "RUN", "SHLS",
        "ARRY", "NOVA", "BE", "BLDP", "FCEL", "CWEN", "DTE", "AES",
        "EIX", "SRE"
    ],
    "Artificial Intelligence": [
        "NVDA", "GOOGL", "MSFT", "AMD", "PLTR", "SNOW", "AI", "CRM", "IBM",
        "AAPL", "ADBE", "MSCI", "DELL", "BIDU", "UPST", "AI", "PATH",
        "SOUN", "VRNT", "ANSS"
    ],
    "Biotechnology": [
        "MRNA", "CRSP", "VRTX", "REGN", "ILMN", "AMGN", "NBIX", "BIIB",
        "INCY", "GILD", "BMRN", "ALNY", "SRPT", "BEAM", "NTLA", "EDIT",
        "BLUE", "SANA", "VKTX", "KRYS"
    ]
}

# Database functions
def setup_stock_database() -> None:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS stocks')
    logger.info("Dropped existing `stocks` table (if it existed).")
    cursor.execute('''
        CREATE TABLE stocks The user says: "No, buy/sell/trim should be based on daily bought_volume and sold_volume but another column to show the deviation logic based on last 5 and 20 days pattern"

This refers to the Streamlit app code for FINRA Short Sale Analysis. The app has tabs for Single Stock, Stock Summary, and Watchlist Summary.

The current logic for signals is:

def get_signal(ratio):

    if ratio > 1.5:

        return 'Buy'

    elif ratio > 1.0:

        return 'Add'

    else:

        return 'Trim'

Where ratio is buy_to_sell_ratio = bought_volume / sold_volume

The user wants:

- Buy/Add/Trim based on daily bought_volume and sold_volume (which it already is, since ratio is from daily).

- But add another column to show deviation logic based on last 5 and 20 days pattern.

From the initial message: "Buy, Add , trim logic should be based on bought_volume and sold_volume. Also, it should analyze deviation in bought and sold volume over the 20 days and show if significant changes. Idea is to understand if something is heavily bought then usual and something heavily sold then usual. For all 3 tabs similar logic used."

So, the signal is still based on daily ratio, but add analysis of deviation over 20 days.

Now, the user is correcting: "No, buy/sell/trim should be based on daily bought_volume and sold_volume but another column to show the deviation logic based on last 5 and 20 days pattern"

It seems "buy/sell/trim" is probably "buy/add/trim".

They want the signal column as is, based on daily.

But add a new column that shows deviation based on last 5 and 20 days.

I need to modify the code to add this new column.

For all 3 tabs.

To implement this, I need to calculate for each symbol the average bought and sold over last 5 days and 20 days, then see the deviation of today's from those averages.

Perhaps compare today's bought_volume to the average of last 5 days (excluding today?), or last 20.

And indicate if it's significantly higher or lower.

The idea is to see if heavily bought than usual.

For each symbol, in the summaries, I need historical data.

In the Single Stock tab, it already fetches data for lookback_days (default 20), so there I have the history.

For the other tabs, currently they only fetch latest day.

So, I need to modify to fetch historical data for the symbols in summaries as well.

But that might be inefficient if many symbols.

There are many themes, many symbols.

But perhaps I can create a function to get historical metrics for a list of symbols.

For efficiency, since in Stock Summary and Watchlist, it's many symbols, but I can cache.

First, I need to define what the deviation column shows.

Perhaps something like: for bought_volume, compare today's to avg of last 20 days.

If today's bought > mean + std or something, "Heavily Bought", etc.

The user said: "analyze deviation in bought and sold volume over the 20 days and show if significant changes."

"if something is heavily bought then usual"

So, perhaps compute z-score or percentage change from average.

For the new column, say "Deviation", and value like "Bought +50% above 20-day avg" or "Sold -30% below 5-day avg" or something.

But to make it simple, perhaps two deviations: vs 5-day avg and vs 20-day avg.

But maybe one.

Also, last 5 and 20 days pattern.

Perhaps use moving averages.

Since it's FINRA data, days are trading days, but FINRA short sale is daily, but weekends no.

The code uses timedelta(days=i), but if weekend, data might not exist.

In process, if no data, skipped.

To implement:

I need a function to get historical data for a symbol, similar to analyze_symbol, but without threshold count.

In analyze_symbol, it fetches for each day separately, which is ok, since FINRA has separate files.

For multiple symbols, if I do for all, it would be many requests.

But since it's Streamlit, run on demand.

For Stock Summary, it has high_buy and high_sell, from all themes.

All symbols are unique? Many overlaps, like SPY in multiple.

symbol_themes maps symbol to first theme.

To add the deviation, I need for each symbol in the df, to have its history.

So, perhaps create a function to compute deviations.

def compute_deviations(symbol, current_bought, current_sold, lookback_days=[5,20]):

    # fetch history

    historical = []

    for i in range(1, 30):  # up to 30 to cover 20 trading days approx

        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")

        data = download_finra_short_sale_data(date)

        if data:

            df = process_finra_short_sale_data(data)

            row = df[df['Symbol'] == symbol]

            if not row.empty:

                r = row.iloc[0]

                total = r['TotalVolume']

                metrics = calculate_metrics(r, total)

                historical.append(metrics)

    # now historical is list of past days

    for lb in lookback_days:

        if len(historical) >= lb:

            past = historical[:lb]

            avg_bought = sum(m['bought_volume'] for m in past) / lb

            avg_sold = sum(m['sold_volume'] for m in past) / lb

            dev_bought = (current_bought - avg_bought) / avg_bought if avg_bought > 0 else 0

            dev_sold = (current_sold - avg_sold) / avg_sold if avg_sold > 0 else 0

            # then format

But for each symbol, fetching 30 days, each day a request to FINRA, and FINRA files are large? But in code, it's txt, probably ok.

But if many symbols, say 100 symbols, 100*30 = 3000 requests, too slow.

No, for each day, I fetch the file once, then query for all symbols.

In current code, for single, it fetches each day separately.

But to optimize, I can fetch each past day once, process df, then for all symbols extract.

For Stock Summary, first get all unique symbols from theme_mapping.

Then, for each past day up to say 30, fetch data, process df, then for each symbol, collect its metrics if present.

Then, store in a dict symbol -> list of past metrics.

Then, for the latest day, have current.

Then, for each, compute avg over last 4 (for 5-day, excluding today), last 19 for 20-day.

Last 5 days means previous 5 trading days.

Since data may miss weekends.

The lookback is number of data points, not calendar days.

In analyze_symbol, lookback_days=20 means last 20 calendar days, but if no data, skipped, so results may have less than 20.

To implement, I can modify get_latest_data to get_historical_data.

But since the current date is July 25, 2025, but that's for simulation?

No, the code is to be executed.

But I need to write the modified code.

Since this is to respond with the updated code.

The task is to update the code based on the user's request.

I need to use the code_execution tool to test or something?

The user message is the code, then the correction.

The human is giving the code, then saying update it.

But in the system, I am Grok, to respond.

But now, the current question is the user's no, ... 

So, I need to provide the updated code.

But to make sure, perhaps use code_execution to run or check.

But since it's Streamlit app, code_execution is Python interpreter, can run code, but Streamlit needs server.

But for syntax, or logic.

Perhaps I can write the code.

First, to add the deviation column.

To avoid many requests, perhaps for Stock Summary and Watchlist, only show daily, and deviation only for Single Stock.

But user said for all 3 tabs.

To optimize, I can cache the historical data.

Since Streamlit has @st.cache_data

I can make a function to get historical for all symbols.

First, get all unique symbols.

all_symbols = set()

for symbols in theme_mapping.values():

    all_symbols.update(symbols)

Then, len(all_symbols) let's see, from the list, many, but say 200-300?

Many duplicates.

Counting roughly, technology 20, financials 20, etc, but overlaps like AMZN in multiple.

Say 300 unique.

For historical, say 20 days, but to fetch 20-30 files.

Each file is fetched once.

Then, for each file, process df = pd.read_csv, delimiter | , then filter len<=4.

Then, for each symbol in all_symbols, if in df, get metrics, append to symbol_history[symbol]

So, time is fetching 30 files, each perhaps few seconds? But in practice, FINRA might be fast.

Then, parsing large file, CNMSshvol is combined national market short volume, probably millions of lines? Wait, is it large?

I think FINRA daily short sale volume is for all stocks, so df has thousands of rows, one per symbol.

Yes, all symbols with short activity.

So, pd.read_csv on a | delimited, say 10k lines, fast.

So, to precompute historical for all, when generating summary.

But in app, when button pressed, with spinner.

Yes, possible.

First, add a function to get_historical_metrics(symbols, lookback_days=30)

returns dict symbol -> list of {'date':, 'bought_volume':, 'sold_volume':, ...} sorted old to new or something.

def get_historical_metrics(symbols: List[str], max_lookback: int = 30) -> dict:

    historical = {symbol: [] for symbol in symbols}

    days_fetched = 0

    i = 1  # start from yesterday

    while days_fetched < max_lookback and i < 60:  # safety

        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")

        data = download_finra_short_sale_data(date_str)

        if data:

            df = process_finra_short_sale_data(data)

            for symbol in symbols:

                symbol_data = df[df['Symbol'] == symbol]

                if not symbol_data.empty:

                    row = symbol_data.iloc[0]

                    total_volume = row.get('TotalVolume', 0)

                    metrics = calculate_metrics(row, total_volume)

                    metrics['date'] = date_str

                    historical[symbol].append(metrics)

            days_fetched += 1

        i += 1

    # sort each by date ascending

    for symbol in historical:

        historical[symbol] = sorted(historical[symbol], key=lambda x: x['date'])

    return historical

Note that days_fetched counts only days with data, so approximately trading days.

For 20 days, set max_lookback=20

For deviation, say for 5-day: take last 5 in historical (which are previous days, since i=1 is yesterday)

Latest is separate.

In get_latest_data, it's for i=0 to 7, find the latest available.

So, latest_date, latest_df

Then, historical is previous days.

For a symbol, past = historical[symbol][-4:] for 5-day (last 4, to make avg of previous 5? No.

Last 5 days pattern, probably average of last 5 previous days.

For today's deviation from recent average.

If historical has previous up to 20 trading days.

len(past) == number of trading days in last ~30 calendar.

To compute 5-day avg: take the most recent 5 past metrics before today.

But today is latest.

So, for deviation, the past does not include today.

In code, when computing for summaries, first get latest_df, latest_date

Then, get historical = get_historical_metrics(all_symbols)

Note that historical starts from i=1, so excludes latest.

In my function, i=1, yesterday, but if today is available, latest is i=0.

But in get_latest_data, it tries i=0 first, if not, i=1, etc.

To avoid duplicate fetch, perhaps modify get_historical to include up to i=0 if available.

But better, fetch all days' data in one loop.

Perhaps a function to get data for last n calendar days, collect per day df if available.

But to simplify.

Let's assume we set max_lookback=30, fetch from i=0 to 30, collect all available.

Then, sort by date descending, first is latest.

Then, for latest, use the most recent.

For historical, the rest.

But in summaries, latest is the most recent available.

To compute avg_5 = average bought of the 2nd to 6th most recent (last 5 before latest)

No, last 5 days pattern means average of last 5 days including today? But deviation of today from past pattern.

To see if today is unusual compared to recent average.

So, compute average of previous days.

For 5-day: avg of last 5 previous trading days.

Similarly for 20.

If have at least 5 previous, take last 5, avg_bought = mean of their bought_volume

Then, dev_bought = (current_bought / avg_bought - 1) * 100 if avg >0 else 0

Similarly for sold.

Then, in the column, perhaps "Deviation" and value like "Bought: +XX%, Sold: -YY% vs 5-day | Bought: ZZ%, Sold: WW% vs 20-day"

Or add two columns: "Dev 5-day", "Dev 20-day"

To show if significant changes.

And perhaps highlight if significant, say >50% deviation.

But user didn't specify threshold.

For now, let's add a column "Bought Dev 5d" , "Bought Dev 20d", "Sold Dev 5d", "Sold Dev 20d"

With values as +XX% or -YY%

If not enough data, "N/A"

That could be.

To save space, perhaps one column "Deviation" with formatted string.

But tables can have more columns.

In code, st.dataframe

Yes.

First, update stock database, but not related.

Now, to implement.

First, in the code, add the get_historical_metrics function.

But modify it to fetch from i=0.

def get_historical_metrics(symbols: List[str], max_days: int = 30) -> dict:

    date_to_df = {}

    for i in range(max_days + 1):

        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")

        data = download_finra_short_sale_data(date_str)

        if data:

            df = process_finra_short_sale_data(data)

            date_to_df[date_str] = df

    # now, for each symbol, collect metrics

    historical = {symbol: [] for symbol in symbols}

    for date_str, df in sorted(date_to_df.items(), reverse=True):  # newest first

        for symbol in symbols:

            symbol_data = df[df['Symbol'] == symbol]

            if not symbol_data.empty:

                row = symbol_data.iloc[0]

                total_volume = row.get('TotalVolume', 0)

                metrics = calculate_metrics(row, total_volume)

                metrics['date'] = date_str

                historical[symbol].append(metrics)

    # but wait, append in newest to oldest

    # better, collect in list, then sort by date ascending

    for symbol in historical:

        historical[symbol] = sorted(historical[symbol], key=lambda x: x['date'])

    return historical

Sorted ascending, old to new, so last is newest.

To find latest_date, max date over all.

But to optimize, perhaps find the latest date where some data.

In current, get_latest_data finds the most recent day with data.

Since we fetch recent first, but anyway.

Now, in generate_stock_summary:

First, symbol_themes as is.

all_symbols = list(set(sum(theme_mapping.values(), [])))  # unique

To get unique.

historical = get_historical_metrics(all_symbols)

# find the latest date: max date over all histories that have data

latest_date = max(max((m['date'] for m in hist), default='0') for hist in historical.values())

if latest_date == '0':

    return ...

# now, for latest metrics

metrics_list = []

for symbol in all_symbols:

    hist = historical[symbol]

    if hist and hist[-1]['date'] == latest_date:

        metrics = hist[-1].copy()

        metrics['Symbol'] = symbol

        metrics_list.append(metrics)

df = pd.DataFrame(metrics_list)

Then, as before, add Theme, etc.

For deviations, add to df.

But df is latest.

For each row, but since symbol, get hist = historical[symbol]

past = hist[:-1]  # exclude latest

# sort? but since sorted ascending, past = hist[-21:-1] for 20, but to take last n previous.

Since ascending, hist[0] old, hist[-1] new.

past = hist[:-1]  # all previous

To compute for 5: if len(past) >=5, avg_bought_5 = sum(p['bought_volume'] for p in past[-5:]) / 5

Similarly for 20: if >=20, avg_bought_20 = sum(past[-20:]) /20

Then, dev_bought_5 = ((current_bought / avg_bought_5 - 1) * 100) if avg_bought_5 >0 else 0

Round to 0.

Then, in display_df, add columns 'Bought Dev 5d' = f"{dev_bought_5:+.0f}%"

Similarly for others.

For sold too.

Yes.

For Single Stock tab, in analyze_symbol, it already has results_df with all days, sorted descending (new to old)

So, for the display_df, which is copy, has multiple rows, each day.

For each row, the "deviation" would be for that day, compared to previous days before it.

But user said for all 3 tabs similar logic.

In single, it's historical table, so perhaps add deviation for each historical day.

To compute, for each day, the avg of previous 5/20 before that day.

Since df is sorted date descending, but to compute, better sort ascending.

I can sort results_df ascending, then for each index, past = df.iloc[:index]

Then, avg_5 = past['bought_volume'].tail(5).mean() if len(past)>=5 else NaN

Then, dev = (df.iloc[index]['bought_volume'] - avg_5) / avg_5 *100 if avg_5>0 else 0

Yes.

But since pandas, can use rolling mean.

For bought_dev_5d = (bought_volume / bought_volume.rolling(5, min_periods=5).mean().shift(1) -1 ) *100

Rolling on previous.

Shift(1) to exclude current.

Yes, perfect.

Sort the df by date ascending, compute rolling_avg_5 = df['bought_volume'].rolling(5, min_periods=1).mean().shift(1)

Min_periods=1, but for deviation, perhaps min 5.

But anyway.

Then, dev_5 = ((df['bought_volume'] / rolling_avg_5) -1 ) *100

Where NaN if rolling_avg NaN.

Then, add column.

But for single stock, since it's one, easy.

For summaries, only for latest day.

For watchlist similar.

Also, in watchlist, selected theme, symbols = theme_mapping[selected_theme]

Then, historical = get_historical_metrics(symbols)

Smaller set.

Yes.

For stock summary, all.

To avoid long time, perhaps set max_days=25 or so.

Also, can cache the historical, but since data updates daily, ttl=3600 or something.

But for now.

Also, in code, update_stock_database(all_symbols) perhaps at start.

But not necessary for this.

Now, to add the columns.

Let's say add 'Bought Dev 5d', 'Bought Dev 20d', 'Sold Dev 5d', 'Sold Dev 20d'

In display_df

First, for summaries.

In generate_stock_summary:

After getting df (latest)

Then, for each symbol in df['Symbol']:

    hist = historical[symbol]

    past = hist[:-1]

    if len(past) >=5:

        avg_b_5 = pd.Series([p['bought_volume'] for p in past[-5:]]).mean()

        avg_s_5 = pd.Series([p['sold_volume'] for p in past[-5:]]).mean()

        dev_b_5 = ((row['bought_volume'] - avg_b_5) / avg_b_5 *100) if avg_b_5 >0 else 0

        Similar for s.

    else:

        dev_b_5 = float('nan')

But since df, I can add columns to display_df

display_df['Bought Dev 5d'] = ''

etc.

No, loop over rows.

Since small, yes.

Better, create lists

dev_b_5_list = []

for _, row in theme_df.iterrows():  # or df in stock summary

    symbol = row['Symbol']

    hist = historical[symbol]

    past = hist[:-1]

    dev_b_5 = 'N/A'

    if len(past) >=5:

        avg = sum(p['bought_volume'] for p in past[-5:]) / 5

        if avg > 0:

            dev = (row['bought_volume'] / avg - 1) * 100

            dev_b_5 = f"{dev:+.0f}%"

    dev_b_5_list.append(dev_b_5)

    # similar for others

Then, display_df['Bought Dev 5d'] = dev_b_5_list

Yes.

Similarly for 20, if len(past)>=20, sum(past[-20:])/20

For Single Stock, since results_df is all days, sorted descending, but to compute rolling.

First, sort asc = results_df.sort_values('date', ascending=True).reset_index(drop=True)

Then, rolling_avg_b_5 = asc['bought_volume'].rolling(5, min_periods=5).mean().shift(1)

dev_b_5 = ((asc['bought_volume'] / rolling_avg_b_5) -1 ) *100

Then, asc['Bought Dev 5d'] = dev_b_5.apply(lambda x: f"{x:+.0f}%" if not pd.isna(x) else "N/A")

Similarly for others.

Then, sort back descending.

display_df = asc.sort_values('date', ascending=False)

Yes.

For signal, still based on daily ratio.

Yes.

The deviation shows if significant changes.

To style, perhaps color if >50% green, <-50% red or something, but not necessary.

I think this is the way.

Now, since current date is July 25, 2025, but for code, irrelevant.

To write the updated code.

But perhaps I need to use tools if needed, but for this, no, just output the code.

The response should be the modified Python code.

Yes.```python
import pandas as pd
import streamlit as st
import requests
import io
from datetime import datetime, timedelta
import plotly.express as px
import logging
import sqlite3
import yfinance as yf
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# Theme mapping for tabs
theme_mapping = {
    "Indexes": [
        "SPY", "QQQ", "IWM", "DIA", "SMH"
    ],
    "Bull Leverage ETF": [
        "SPXL", "UPRO", "TQQQ", "SOXL", "UDOW", "FAS", "SPUU", "TNA"
    ],
    "Bear Leverage ETF": [
        "SQQQ", "SPXS", "SOXS", "SDOW", "FAZ","SPDN", "TZA", "SPXU"
    ],
    "Volatility": [
        "VXX", "VIXY", "UVXY"
    ],
    "Bonds": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "AGG"
    ],
    "Commodities": [
        "SPY", "GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"
    ],
    "Nuclear Power": [
        "CEG", "NNE", "GEV", "OKLO", "UUUU", "ASPI", "CCJ"
    ],
    "Crypto": [
        "IBIT", "FBTC", "MSTR", "COIN", "HOOD", "ETHU"
    ],
    "Metals": [
        "GLD", "SLV", "GDX", "GDXJ", "IAU", "SIVR"
    ],
    "Real Estate": [
        "VNQ", "IYR", "XHB", "XLF", "SPG", "PLD", "AMT", "DLR"
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "NKE", "MCD", "DIS", "LOW", "TGT", "LULU"
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "CL", "KMB", "MDLZ", "GIS"
    ],
    "Utilities": [
        "XLU", "DUK", "SO", "D", "NEE", "EXC", "AEP", "SRE", "ED"
    ],
    "Telecommunications": [
        "XLC", "T", "VZ", "TMUS", "S", "LUMN", "VOD"
    ],
    "Materials": [
        "XLB", "XME", "XLI", "FCX", "NUE", "DD", "APD", "LIN", "IFF"
    ],
    "Transportation": [
        "UPS", "FDX", "DAL", "UAL", "LUV", "CSX", "NSC", "KSU", "WAB"
    ],
    "Aerospace & Defense": [
        "LMT", "BA", "NOC", "RTX", "GD", "HII", "LHX", "COL", "TXT"
    ],
    "Retail": [
        "AMZN", "WMT", "TGT", "COST", "HD", "LOW", "TJX", "M", "KSS"
    ],
    "Automotive": [
        "TSLA", "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "BYDDF", "FCAU"
    ],
    "Pharmaceuticals": [
        "PFE", "MRK", "JNJ", "ABBV", "BMY", "GILD", "AMGN", "LLY", "VRTX"
    ],
    "Biotechnology": [
        "AMGN", "REGN", "ILMN", "VRTX", "CRSP", "MRNA", "BMRN", "ALNY",
        "SRPT", "EDIT", "NTLA", "BEAM", "BLUE", "FATE", "SANA"
    ],
    "Insurance": [
        "AIG", "PRU", "MET", "UNM", "LNC", "TRV", "CINF", "PGR", "ALL"
    ],
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
        "ORCL", "CRM", "ADBE", "INTC", "CSCO", "QCOM", "TXN", "IBM",
        "NOW", "AVGO", "INTU", "PANW", "SNOW"
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW",
        "COF", "MET", "AIG", "BK", "BLK", "TFC", "USB", "PNC", "CME", "SPGI"
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV", "TMO", "AMGN", "GILD",
        "CVS", "MDT", "BMY", "ABT", "DHR", "ISRG", "SYK", "REGN", "VRTX",
        "CI", "ZTS"
    ],
    "Consumer": [
        "WMT", "PG", "KO", "PEP", "COST", "MCD", "DIS", "NKE", "SBUX",
        "LOW", "TGT", "HD", "CL", "MO", "KHC", "PM", "TJX", "DG", "DLTR", "YUM"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "VLO",
        "XLE", "HES", "WMB", "KMI", "OKE", "HAL", "BKR", "FANG", "DVN",
        "TRGP", "APA"
    ],
    "Industrials": [
        "CAT", "DE", "UPS", "FDX", "BA", "HON", "UNP", "MMM", "GE", "LMT",
        "RTX", "GD", "CSX", "NSC", "WM", "ETN", "ITW", "EMR", "PH", "ROK"
    ],
    "Semiconductors": [
        "NVDA", "AMD", "QCOM", "TXN", "INTC", "AVGO", "ASML", "KLAC",
        "LRCX", "AMAT", "ADI", "MCHP", "ON", "STM", "MPWR", "TER", "ENTG",
        "SWKS", "QRVO", "LSCC"
    ],
    "Cybersecurity": [
        "CRWD", "PANW", "ZS", "FTNT", "S", "OKTA", "CYBR", "RPD", "NET",
        "QLYS", "TENB", "VRNS", "SPLK", "CHKP", "FEYE", "DDOG", "ESTC",
        "FSLY", "MIME", "KNBE"
    ],
    "Quantum Computing": [
        "IBM", "GOOGL", "MSFT", "RGTI", "IONQ", "QUBT", "HON", "QCOM",
        "INTC", "AMAT", "MKSI", "NTNX", "XERI", "QTUM", "FORM",
        "LMT", "BA", "NOC", "ACN"
    ],
    "Clean Energy": [
        "TSLA", "ENPH", "FSLR", "NEE", "PLUG", "SEDG", "RUN", "SHLS",
        "ARRY", "NOVA", "BE", "BLDP", "FCEL", "CWEN", "DTE", "AES",
        "EIX", "SRE"
    ],
    "Artificial Intelligence": [
        "NVDA", "GOOGL", "MSFT", "AMD", "PLTR", "SNOW", "AI", "CRM", "IBM",
        "AAPL", "ADBE", "MSCI", "DELL", "BIDU", "UPST", "AI", "PATH",
        "SOUN", "VRNT", "ANSS"
    ],
    "Biotechnology": [
        "MRNA", "CRSP", "VRTX", "REGN", "ILMN", "AMGN", "NBIX", "BIIB",
        "INCY", "GILD", "BMRN", "ALNY", "SRPT", "BEAM", "NTLA", "EDIT",
        "BLUE", "SANA", "VKTX", "KRYS"
    ]
}

# Database functions
def setup_stock_database() -> None:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS stocks')
    logger.info("Dropped existing `stocks` table (if it existed).")
    cursor.execute('''
        CREATE TABLE stocks (
            symbol TEXT PRIMARY KEY,
            price REAL,
            market_cap REAL,
            last_updated TEXT
        )
    ''')
    logger.info("Created `stocks` table with correct schema.")
    conn.commit()
    conn.close()

def check_and_setup_database() -> None:
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'")
        if not cursor.fetchone():
            conn.close()
            setup_stock_database()
            return
        cursor.execute("PRAGMA table_info(stocks)")
        columns = [info[1] for info in cursor.fetchall()]
        if not all(col in columns for col in ['symbol', 'price', 'market_cap', 'last_updated']):
            conn.close()
            setup_stock_database()
            return
        conn.close()
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        setup_stock_database()

check_and_setup_database()

def update_stock_database(symbols: List[str]) -> None:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    try:
        tickers = yf.Tickers(' '.join(symbols))
        for symbol in symbols:
            try:
                ticker = tickers.tickers[symbol]
                info = ticker.info
                price = info.get('regularMarketPrice', 0)
                market_cap = info.get('marketCap', 0)
                cursor.execute('''
                    INSERT OR REPLACE INTO stocks (symbol, price, market_cap, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, price, market_cap, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                cursor.execute('''
                    INSERT OR REPLACE INTO stocks (symbol, price, market_cap, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, 0, 0, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
    finally:
        conn.commit()
        conn.close()

def get_stock_info_from_db(symbol: str) -> dict:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT price, market_cap FROM stocks WHERE symbol = ?', (symbol,))
    result = cursor.fetchone()
    conn.close()
    return {'price': result[0], 'market_cap': result[1]} if result else {'price': 0, 'market_cap': 0}

# FINRA data processing functions
def download_finra_short_sale_data(date: str) -> Optional[str]:
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def process_finra_short_sale_data(data: Optional[str]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(data), delimiter="|")
    return df[df["Symbol"].str.len() <= 4]

def calculate_metrics(row: pd.Series, total_volume: float) -> dict:
    short_volume = row.get('ShortVolume', 0)
    short_exempt_volume = row.get('ShortExemptVolume', 0)
    bought_volume = short_volume + short_exempt_volume
    sold_volume = total_volume - bought_volume
    buy_to_sell_ratio = bought_volume / sold_volume if sold_volume > 0 else float('inf')
    short_volume_ratio = bought_volume / total_volume if total_volume > 0 else 0
    return {
        'total_volume': total_volume,
        'bought_volume': bought_volume,
        'sold_volume': sold_volume,
        'buy_to_sell_ratio': round(buy_to_sell_ratio, 2),
        'short_volume_ratio': round(short_volume_ratio, 4)
    }

# Historical metrics function
@st.cache_data(ttl=3600)
def get_historical_metrics(symbols: List[str], max_days: int = 30) -> dict:
    date_to_df = {}
    for i in range(max_days):
        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date_str)
        if data:
            df = process_finra_short_sale_data(data)
            date_to_df[date_str] = df
    historical = {symbol: [] for symbol in symbols}
    for date_str, df in date_to_df.items():
        for symbol in symbols:
            symbol_data = df[df['Symbol'] == symbol]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                metrics['date'] = pd.to_datetime(date_str, format='%Y%m%d')
                historical[symbol].append(metrics)
    for symbol in historical:
        historical[symbol] = sorted(historical[symbol], key=lambda x: x['date'])
    return historical

# Single stock analysis
@st.cache_data(ttl=11520)
def analyze_symbol(symbol: str, lookback_days: int = 20, threshold: float = 1.5) -> tuple[pd.DataFrame, int]:
    results = []
    significant_days = 0
    cumulative_bought = 0
    cumulative_sold = 0
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            symbol_data = df[df['Symbol'] == symbol]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                cumulative_bought += metrics['bought_volume']
                cumulative_sold += metrics['sold_volume']
                if metrics['buy_to_sell_ratio'] > threshold:
                    significant_days += 1
                metrics['date'] = date
                metrics['cumulative_bought'] = cumulative_bought
                metrics['cumulative_sold'] = cumulative_sold
                results.append(metrics)
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
        df_results = df_results.sort_values('date', ascending=True)
        # Compute deviation
        df_results['prev_avg_bought_5'] = df_results['bought_volume'].rolling(window=5, min_periods=5).mean().shift(1)
        df_results['prev_avg_sold_5'] = df_results['sold_volume'].rolling(window=5, min_periods=5).mean().shift(1)
        df_results['prev_avg_bought_20'] = df_results['bought_volume'].rolling(window=20, min_periods=20).mean().shift(1)
        df_results['prev_avg_sold_5'] = df_results['sold_volume'].rolling(window=5, min_periods=5).mean().shift(1)
        df_results['dev_bought_5'] = df_results.apply(lambda r: ((r['bought_volume'] - r['prev_avg_bought_5']) / r['prev_avg_bought_5'] * 100) if pd.notnull(r['prev_avg_bought_5']) and r['prev_avg_bought_5'] > 0 else np.nan, axis=1).round(0)
        df_results['dev_sold_5'] = df_results.apply(lambda r: ((r['sold_volume'] - r['prev_avg_sold_5']) / r['prev_avg_sold_5'] * 100) if pd.notnull(r['prev_avg_sold_5']) and r['prev_avg_sold_5'] > 0 else np.nan, axis=1).round(0)
        df_results['dev_bought_20'] = df_results.apply(lambda r: ((r['bought_volume'] - r['prev_avg_bought_20']) / r['prev_avg_bought_20'] * 100) if pd.notnull(r['prev_avg_bought_20']) and r['prev_avg_bought_20'] > 0 else np.nan, axis=1).round(0)
        df_results['dev_sold_20'] = df_results.apply(lambda r: ((r['sold_volume'] - r['prev_avg_sold_20']) / r['prev_avg_sold_20'] * 100) if pd.notnull(r['prev_avg_sold_20']) and r['prev_avg_sold_20'] > 0 else np.nan, axis=1).round(0)
        df_results = df_results.sort_values('date', ascending=False)
    return df_results, significant_days

# Latest data fetch
def get_latest_data(symbols: List[str] = None) -> tuple[pd.DataFrame, Optional[str]]:
    for i in range(7):  # Check the last 7 days
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            if not df.empty:
                if symbols:
                    df = df[df['Symbol'].isin(symbols)]
                return df, date
    return pd.DataFrame(), None

# Stock summary generation
def generate_stock_summary() -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    symbol_themes = {}
    for theme, symbols in theme_mapping.items():
        for symbol in symbols:
            if symbol not in symbol_themes:
                symbol_themes[symbol] = theme
    all_symbols = list(set([s for symbols in theme_mapping.values() for s in symbols]))
    historical = get_historical_metrics(all_symbols)
    latest_date = None
    for hist in historical.values():
        if hist:
            latest_date = max(latest_date, hist[-1]['date'].strftime('%Y%m%d')) if latest_date else hist[-1]['date'].strftime('%Y%m%d')
    if not latest_date:
        return pd.DataFrame(), pd.DataFrame(), None
    metrics_list = []
    for symbol in all_symbols:
        hist = historical[symbol]
        if hist and hist[-1]['date'].strftime('%Y%m%d') == latest_date:
            metrics = hist[-1].copy()
            metrics['Symbol'] = symbol
            past = hist[:-1]
            dev_bought_5 = np.nan
            dev_sold_5 = np.nan
            dev_bought_20 = np.nan
            dev_sold_20 = np.nan
            if len(past) >= 5:
                avg_bought_5 = np.mean([p['bought_volume'] for p in past[-5:]])
                avg_sold_5 = np.mean([p['sold_volume'] for p in past[-5:]])
                if avg_bought_5 > 0:
                    dev_bought_5 = ((metrics['bought_volume'] - avg_bought_5) / avg_bought_5 * 100).round(0)
                if avg_sold_5 > 0:
                    dev_sold_5 = ((metrics['sold_volume'] - avg_sold_5) / avg_sold_5 * 100).round(0)
            if len(past) >= 20:
                avg_bought_20 = np.mean([p['bought_volume'] for p in past[-20:]])
                avg_sold_20 = np.mean([p['sold_volume'] for p in past[-20:]])
                if avg_bought_20 > 0:
                    dev_bought_20 = ((metrics['bought_volume'] - avg_bought_20) / avg_bought_20 * 100).round(0)
                if avg_sold_20 > 0:
                    dev_sold_20 = ((metrics['sold_volume'] - avg_sold_20) / avg_sold_20 * 100).round(0)
            metrics['dev_bought_5'] = dev_bought_5
            metrics['dev_sold_5'] = dev_sold_5
            metrics['dev_bought_20'] = dev_bought_20
            metrics['dev_sold_20'] = dev_sold_20
            metrics_list.append(metrics)
    df = pd.DataFrame(metrics_list)
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    df['Theme'] = df['Symbol'].map(symbol_themes)
    high_buy_df = df[df['bought_volume'] > 2 * df['sold_volume']].copy()
    high_sell_df = df[df['sold_volume'] > 2 * df['bought_volume']].copy()
    columns = ['Symbol', 'Theme', 'buy_to_sell_ratio', 'bought_volume', 'sold_volume', 'total_volume']
    if not high_buy_df.empty:
        high_buy_df = high_buy_df[columns].sort_values(by='buy_to_sell_ratio', ascending=False)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            high_buy_df[col] = high_buy_df[col].astype(int)
        high_buy_df['buy_to_sell_ratio'] = high_buy_df['buy_to_sell_ratio'].round(2)
    if not high_sell_df.empty:
        high_sell_df = high_sell_df[columns].sort_values(by='buy_to_sell_ratio', ascending=True)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            high_sell_df[col] = high_sell_df[col].astype(int)
        high_sell_df['buy_to_sell_ratio'] = high_sell_df['buy_to_sell_ratio'].round(2)
    return high_buy_df, high_sell_df, latest_date

def get_signal(ratio):
    if ratio > 1.5:
        return 'Buy'
    elif ratio > 1.0:
        return 'Add'
    else:
        return 'Trim'

def style_signal(val):
    color = 'green' if val in ['Buy', 'Add'] else 'red'
    return f'color: {color}; font-weight: bold'

def style_deviation(val):
    if pd.isna(val):
        return ''
    try:
        num = float(val.rstrip('%'))
        if num > 50:
            return 'color: green; font-weight: bold'
        if num < -50:
            return 'color: red; font-weight: bold'
    except:
        pass
    return ''

def run():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
            padding: 10px;
        }
        .stSelectbox {
            max-width: 300px;
        }
        .stDataFrame {
            font-size: 14px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("FINRA Short Sale Analysis")
    
    # Create tabs
    tabs = st.tabs(["Single Stock", "Stock Summary", "Watchlist Summary"])
    
    # Single Stock Tab
    with tabs[0]:
        st.subheader("Single Stock Analysis")
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Enter Symbol", "NVDA").strip().upper()
            lookback_days = st.slider("Lookback Days", 1, 30, 20)
        with col2:
            threshold = st.number_input("Buy/Sell Ratio Threshold", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        if st.button("Analyze Stock"):
            with st.spinner(f"Analyzing {symbol}..."):
                results_df, significant_days = analyze_symbol(symbol, lookback_days, threshold)
                if not results_df.empty:
                    avg_buy = results_df['bought_volume'].mean()
                    avg_sell = results_df['sold_volume'].mean()
                    avg_total = results_df['total_volume'].mean()
                    total_buy = results_df['bought_volume'].sum()
                    total_sell = results_df['sold_volume'].sum()
                    total_volume_sum = results_df['total_volume'].sum()
                    aggregate_ratio = total_buy / total_sell if total_sell > 0 else float('inf')
                    
                    st.subheader("Summary Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Bought", f"{total_buy:,.0f}")
                    col2.metric("Total Sold", f"{total_sell:,.0f}")
                    col3.metric("Avg Buy Volume", f"{avg_buy:,.0f}")
                    col4.metric("Avg Sell Volume", f"{avg_sell:,.0f}")
                    col1.metric("Total Volume", f"{total_volume_sum:,.0f}")
                    col2.metric("Aggregate Buy Ratio", f"{aggregate_ratio:.2f}")
                    
                    display_df = results_df.copy()
                    display_df['Short %'] = (display_df['bought_volume'] / display_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    display_df['BOT %'] = display_df['Short %']
                    display_df['Signal'] = display_df['buy_to_sell_ratio'].apply(get_signal)
                    display_df['Bought Dev 5d'] = display_df['dev_bought_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Sold Dev 5d'] = display_df['dev_sold_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Bought Dev 20d'] = display_df['dev_bought_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Sold Dev 20d'] = display_df['dev_sold_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        display_df[col] = display_df[col].astype(int).apply(lambda x: f"{x:,.0f}")
                    display_df['buy_to_sell_ratio'] = display_df['buy_to_sell_ratio'].round(2)
                    display_df['date'] = display_df['date'].dt.strftime('%Y%m%d')
                    columns = ['date', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d', 'Signal']
                    st.dataframe(display_df[columns].style.applymap(style_signal, subset=['Signal']).applymap(style_deviation, subset=['Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d']))
                else:
                    st.write(f"No data available for {symbol}.")
    
    # Stock Summary Tab
    with tabs[1]:
        st.subheader("Stock Summary")
        if st.button("Generate Stock Summary"):
            with st.spinner("Analyzing stock volume data..."):
                high_buy_df, high_sell_df, latest_date = generate_stock_summary()
                if latest_date:
                    st.write(f"Data analyzed for: {latest_date}")
                st.write("### High Buy Stocks (Bought > 2x Sold)")
                if not high_buy_df.empty:
                    high_buy_df['Short %'] = (high_buy_df['bought_volume'] / high_buy_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    high_buy_df['BOT %'] = high_buy_df['Short %']
                    high_buy_df['Signal'] = high_buy_df['buy_to_sell_ratio'].apply(get_signal)
                    high_buy_df['Bought Dev 5d'] = high_buy_df['dev_bought_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_buy_df['Sold Dev 5d'] = high_buy_df['dev_sold_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_buy_df['Bought Dev 20d'] = high_buy_df['dev_bought_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_buy_df['Sold Dev 20d'] = high_buy_df['dev_sold_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        high_buy_df[col] = high_buy_df[col].apply(lambda x: f"{x:,.0f}")
                    columns = ['Symbol', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d', 'Signal']
                    st.dataframe(high_buy_df[columns].style.applymap(style_signal, subset=['Signal']).applymap(style_deviation, subset=['Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d']))
                else:
                    st.write("No high buy stocks found.")
                st.write("### High Sell Stocks (Sold > 2x Bought)")
                if not high_sell_df.empty:
                    high_sell_df['Short %'] = (high_sell_df['bought_volume'] / high_sell_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    high_sell_df['BOT %'] = high_sell_df['Short %']
                    high_sell_df['Signal'] = high_sell_df['buy_to_sell_ratio'].apply(get_signal)
                    high_sell_df['Bought Dev 5d'] = high_sell_df['dev_bought_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_sell_df['Sold Dev 5d'] = high_sell_df['dev_sold_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_sell_df['Bought Dev 20d'] = high_sell_df['dev_bought_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_sell_df['Sold Dev 20d'] = high_sell_df['dev_sold_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        high_sell_df[col] = high_sell_df[col].apply(lambda x: f"{x:,.0f}")
                    columns = ['Symbol', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d', 'Signal']
                    st.dataframe(high_sell_df[columns].style.applymap(style_signal, subset=['Signal']).applymap(style_deviation, subset=['Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d']))
                else:
                    st.write("No high sell stocks found.")
    
    # Watchlist Summary Tab
    with tabs[2]:
        st.subheader("Watchlist Summary")
        selected_theme = st.selectbox("Select Watchlist (Theme)", list(theme_mapping.keys()), index=0)
        if st.button("Generate Watchlist Summary"):
            with st.spinner(f"Analyzing {selected_theme}..."):
                symbols = theme_mapping[selected_theme]
                historical = get_historical_metrics(symbols)
                latest_date = None
                for hist in historical.values():
                    if hist:
                        latest_date = max(latest_date, hist[-1]['date'].strftime('%Y%m%d_database()
            return
        conn.close()
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        setup_stock_database()

check_and_setup_database()

def update_stock_database(symbols: List[str]) -> None:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    try:
        tickers = yf.Tickers(' '.join(symbols))
        for symbol in symbols:
            try:
                ticker = tickers.tickers[symbol]
                info = ticker.info
                price = info.get('regularMarketPrice', 0)
                market_cap = info.get('marketCap', 0)
                cursor.execute('''
                    INSERT OR REPLACE INTO stocks (symbol, price, market_cap, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, price, market_cap, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                cursor.execute('''
                    INSERT OR REPLACE INTO stocks (symbol, price, market_cap, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, 0, 0, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
    finally:
        conn.commit()
        conn.close()

def get_stock_info_from_db(symbol: str) -> dict:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT price, market_cap FROM stocks WHERE symbol = ?', (symbol,))
    result = cursor.fetchone()
    conn.close()
    return {'price': result[0], 'market_cap': result[1]} if result else {'price': 0, 'market_cap': 0}

# FINRA data processing functions
def download_finra_short_sale_data(date: str) -> Optional[str]:
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def process_finra_short_sale_data(data: Optional[str]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(data), delimiter="|")
    return df[df["Symbol"].str.len() <= 4]

def calculate_metrics(row: pd.Series, total_volume: float) -> dict:
    short_volume = row.get('ShortVolume', 0)
    short_exempt_volume = row.get('ShortExemptVolume', 0)
    bought_volume = short_volume + short_exempt_volume
    sold_volume = total_volume - bought_volume
    buy_to_sell_ratio = bought_volume / sold_volume if sold_volume > 0 else float('inf')
    short_volume_ratio = bought_volume / total_volume if total_volume > 0 else 0
    return {
        'total_volume': total_volume,
        'bought_volume': bought_volume,
        'sold_volume': sold_volume,
        'buy_to_sell_ratio': round(buy_to_sell_ratio, 2),
        'short_volume_ratio': round(short_volume_ratio, 4)
    }

# Historical data fetch
def get_historical_metrics(symbols: List[str], max_days: int = 30) -> dict:
    date_to_df = {}
    for i in range(max_days + 1):
        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date_str)
        if data:
            df = process_finra_short_sale_data(data)
            date_to_df[date_str] = df

    historical = {symbol: [] for symbol in symbols}
    for date_str, df in date_to_df.items():
        for symbol in symbols:
            symbol_data = df[df['Symbol'] == symbol]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                metrics['date'] = pd.to_datetime(date_str, format='%Y%m%d')
                historical[symbol].append(metrics)

    for symbol in historical:
        historical[symbol] = sorted(historical[symbol], key=lambda x: x['date'])

    return historical

# Single stock analysis
@st.cache_data(ttl=11520)
def analyze_symbol(symbol: str, lookback_days: int = 20, threshold: float = 1.5) -> tuple[pd.DataFrame, int]:
    results = []
    significant_days = 0
    cumulative_bought = 0
    cumulative_sold = 0
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            symbol_data = df[df['Symbol'] == symbol]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                cumulative_bought += metrics['bought_volume']
                cumulative_sold += metrics['sold_volume']
                if metrics['buy_to_sell_ratio'] > threshold:
                    significant_days += 1
                metrics['date'] = date
                metrics['cumulative_bought'] = cumulative_bought
                metrics['cumulative_sold'] = cumulative_sold
                results.append(metrics)
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
        df_asc = df_results.sort_values('date', ascending=True).reset_index(drop=True)
        df_asc['rolling_avg_b_5'] = df_asc['bought_volume'].rolling(5, min_periods=5).mean().shift(1)
        df_asc['rolling_avg_s_5'] = df_asc['sold_volume'].rolling(5, min_periods=5).mean().shift(1)
        df_asc['rolling_avg_b_20'] = df_asc['bought_volume'].rolling(20, min_periods=20).mean().shift(1)
        df_asc['rolling_avg_s_20'] = df_asc['sold_volume'].rolling(20, min_periods=20).mean().shift(1)
        df_asc['dev_b_5'] = ((df_asc['bought_volume'] - df_asc['rolling_avg_b_5']) / df_asc['rolling_avg_b_5'] * 100).round(0) if 'rolling_avg_b_5' in df_asc else np.nan
        df_asc['dev_s_5'] = ((df_asc['sold_volume'] - df_asc['rolling_avg_s_5']) / df_asc['rolling_avg_s_5'] * 100).round(0)
        df_asc['dev_b_20'] = ((df_asc['bought_volume'] - df_asc['rolling_avg_b_20']) / df_asc['rolling_avg_b_20'] * 100).round(0)
        df_asc['dev_s_20'] = ((df_asc['sold_volume'] - df_asc['rolling_avg_s_20']) / df_asc['rolling_avg_s_20'] * 100).round(0)
        df_results = df_asc.sort_values('date', ascending=False)
    return df_results, significant_days

# Latest data fetch
def get_latest_data(symbols: List[str] = None) -> tuple[pd.DataFrame, Optional[str]]:
    for i in range(7):  # Check the last 7 days
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            if not df.empty:
                if symbols:
                    df = df[df['Symbol'].isin(symbols)]
                return df, date
    return pd.DataFrame(), None

# Stock summary generation
def generate_stock_summary() -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    symbol_themes = {}
    for theme, symbols in theme_mapping.items():
        for symbol in symbols:
            if symbol not in symbol_themes:
                symbol_themes[symbol] = theme
    all_symbols = list(set(symbol_themes.keys()))
    historical = get_historical_metrics(all_symbols)
    latest_date = max(max((m['date'].strftime('%Y%m%d') for m in hist), default=None) for hist in historical.values() if hist)
    if not latest_date:
        return pd.DataFrame(), pd.DataFrame(), None
    metrics_list = []
    for symbol in all_symbols:
        hist = historical[symbol]
        if hist and hist[-1]['date'].strftime('%Y%m%d') == latest_date:
            metrics = hist[-1].copy()
            metrics['Symbol'] = symbol
            past = hist[:-1]
            dev_b_5 = np.nan
            dev_s_5 = np.nan
            dev_b_20 = np.nan
            dev_s_20 = np.nan
            if len(past) >= 5:
                avg_b_5 = pd.Series([p['bought_volume'] for p in past[-5:]]).mean()
                avg_s_5 = pd.Series([p['sold_volume'] for p in past[-5:]]).mean()
                dev_b_5 = ((metrics['bought_volume'] - avg_b_5) / avg_b_5 * 100).round(0) if avg_b_5 > 0 else np.nan
                dev_s_5 = ((metrics['sold_volume'] - avg_s_5) / avg_s_5 * 100).round(0) if avg_s_5 > 0 else np.nan
            if len(past) >= 20:
                avg_b_20 = pd.Series([p['bought_volume'] for p in past[-20:]]).mean()
                avg_s_20 = pd.Series([p['sold_volume'] for p in past[-20:]]).mean()
                dev_b_20 = ((metrics['bought_volume'] - avg_b_20) / avg_b_20 * 100).round(0) if avg_b_20 > 0 else np.nan
                dev_s_20 = ((metrics['sold_volume'] - avg_s_20) / avg_s_20 * 100).round(0) if avg_s_20 > 0 else np.nan
            metrics['dev_b_5'] = dev_b_5
            metrics['dev_s_5'] = dev_s_5
            metrics['dev_b_20'] = dev_b_20
            metrics['dev_s_20'] = dev_s_20
            metrics_list.append(metrics)
    df = pd.DataFrame(metrics_list)
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    df['Theme'] = df['Symbol'].map(symbol_themes)
    high_buy_df = df[df['bought_volume'] > 2 * df['sold_volume']].copy()
    high_sell_df = df[df['sold_volume'] > 2 * df['bought_volume']].copy()
    columns = ['Symbol', 'Theme', 'buy_to_sell_ratio', 'bought_volume', 'sold_volume', 'total_volume']
    if not high_buy_df.empty:
        high_buy_df = high_buy_df[columns].sort_values(by='buy_to_sell_ratio', ascending=False)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            high_buy_df[col] = high_buy_df[col].astype(int)
        high_buy_df['buy_to_sell_ratio'] = high_buy_df['buy_to_sell_ratio'].round(2)
    if not high_sell_df.empty:
        high_sell_df = high_sell_df[columns].sort_values(by='buy_to_sell_ratio', ascending=True)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            high_sell_df[col] = high_sell_df[col].astype(int)
        high_sell_df['buy_to_sell_ratio'] = high_sell_df['buy_to_sell_ratio'].round(2)
    return high_buy_df, high_sell_df, latest_date

def get_signal(ratio):
    if ratio > 1.5:
        return 'Buy'
    elif ratio > 1.0:
        return 'Add'
    else:
        return 'Trim'

def style_signal(val):
    color = 'green' if val in ['Buy', 'Add'] else 'red'
    return f'color: {color}; font-weight: bold'

def style_dev(val):
    if val == '-':
        return ''
    num = float(val.rstrip('%'))
    color = 'green' if num > 0 else 'red'
    return f'color: {color}; font-weight: bold' if abs(num) > 50 else ''

def run():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
            padding: 10px;
        }
        .stSelectbox {
            max-width: 300px;
        }
        .stDataFrame {
            font-size: 14px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("FINRA Short Sale Analysis")
    
    # Create tabs
    tabs = st.tabs(["Single Stock", "Stock Summary", "Watchlist Summary"])
    
    # Single Stock Tab
    with tabs[0]:
        st.subheader("Single Stock Analysis")
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Enter Symbol", "NVDA").strip().upper()
            lookback_days = st.slider("Lookback Days", 1, 30, 20)
        with col2:
            threshold = st.number_input("Buy/Sell Ratio Threshold", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        if st.button("Analyze Stock"):
            with st.spinner(f"Analyzing {symbol}..."):
                results_df, significant_days = analyze_symbol(symbol, lookback_days, threshold)
                if not results_df.empty:
                    avg_buy = results_df['bought_volume'].mean()
                    avg_sell = results_df['sold_volume'].mean()
                    avg_total = results_df['total_volume'].mean()
                    total_buy = results_df['bought_volume'].sum()
                    total_sell = results_df['sold_volume'].sum()
                    total_volume_sum = results_df['total_volume'].sum()
                    aggregate_ratio = total_buy / total_sell if total_sell > 0 else float('inf')
                    
                    st.subheader("Summary Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Bought", f"{total_buy:,.0f}")
                    col2.metric("Total Sold", f"{total_sell:,.0f}")
                    col3.metric("Avg Buy Volume", f"{avg_buy:,.0f}")
                    col4.metric("Avg Sell Volume", f"{avg_sell:,.0f}")
                    col1.metric("Total Volume", f"{total_volume_sum:,.0f}")
                    col2.metric("Aggregate Buy Ratio", f"{aggregate_ratio:.2f}")
                    
                    display_df = results_df.copy()
                    display_df['Short %'] = (display_df['bought_volume'] / display_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    display_df['BOT %'] = display_df['Short %']
                    display_df['Signal'] = display_df['buy_to_sell_ratio'].apply(get_signal)
                    display_df['Bought Dev 5d'] = display_df['dev_b_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Sold Dev 5d'] = display_df['dev_s_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Bought Dev 20d'] = display_df['dev_b_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Sold Dev 20d'] = display_df['dev_s_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        display_df[col] = display_df[col].astype(int).apply(lambda x: f"{x:,.0f}")
                    display_df['buy_to_sell_ratio'] = display_df['buy_to_sell_ratio'].round(2)
                    display_df['date'] = display_df['date'].dt.strftime('%Y%m%d')
                    columns = ['date', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d', 'Signal']
                    st.dataframe(display_df[columns].style.applymap(style_signal, subset=['Signal']).applymap(style_dev, subset=['Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d']))
                else:
                    st.write(f"No data available for {symbol}.")
    
    # Stock Summary Tab
    with tabs[1]:
        st.subheader("Stock Summary")
        if st.button("Generate Stock Summary"):
            with st.spinner("Analyzing stock volume data..."):
                high_buy_df, high_sell_df, latest_date = generate_stock_summary()
                if latest_date:
                    st.write(f"Data analyzed for: {latest_date}")
                st.write("### High Buy Stocks (Bought > 2x Sold)")
                if not high_buy_df.empty:
                    high_buy_df['Short %'] = (high_buy_df['bought_volume'] / high_buy_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    high_buy_df['BOT %'] = high_buy_df['Short %']
                    high_buy_df['Signal'] = high_buy_df['buy_to_sell_ratio'].apply(get_signal)
                    high_buy_df['Bought Dev 5d'] = high_buy_df['dev_b_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_buy_df['Sold Dev 5d'] = high_buy_df['dev_s_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_buy_df['Bought Dev 20d'] = high_buy_df['dev_b_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_buy_df['Sold Dev 20d'] = high_buy_df['dev_s_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        high_buy_df[col] = high_buy_df[col].apply(lambda x: f"{x:,.0f}")
                    columns = ['Symbol', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d', 'Signal']
                    st.dataframe(high_buy_df[columns].style.applymap(style_signal, subset=['Signal']).applymap(style_dev, subset=['Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d']))
                else:
                    st.write("No high buy stocks found.")
                st.write("### High Sell Stocks (Sold > 2x Bought)")
                if not high_sell_df.empty:
                    high_sell_df['Short %'] = (high_sell_df['bought_volume'] / high_sell_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    high_sell_df['BOT %'] = high_sell_df['Short %']
                    high_sell_df['Signal'] = high_sell_df['buy_to_sell_ratio'].apply(get_signal)
                    high_sell_df['Bought Dev 5d'] = high_sell_df['dev_b_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_sell_df['Sold Dev 5d'] = high_sell_df['dev_s_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_sell_df['Bought Dev 20d'] = high_sell_df['dev_b_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    high_sell_df['Sold Dev 20d'] = high_sell_df['dev_s_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        high_sell_df[col] = high_sell_df[col].apply(lambda x: f"{x:,.0f}")
                    columns = ['Symbol', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d', 'Signal']
                    st.dataframe(high_sell_df[columns].style.applymap(style_signal, subset=['Signal']).applymap(style_dev, subset=['Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d']))
                else:
                    st.write("No high sell stocks found.")
    
    # Watchlist Summary Tab
    with tabs[2]:
        st.subheader("Watchlist Summary")
        selected_theme = st.selectbox("Select Watchlist (Theme)", list(theme_mapping.keys()), index=0)
        if st.button("Generate Watchlist Summary"):
            with st.spinner(f"Analyzing {selected_theme}..."):
                symbols = theme_mapping[selected_theme]
                historical = get_historical_metrics(symbols)
                latest_date = max(max((m['date'].strftime('%Y%m%d') for m in hist), default=None) for hist in historical.values() if hist)
                if latest_date:
                    st.write(f"Data for: {latest_date}")
                    metrics_list = []
                    for symbol in symbols:
                        hist = historical[symbol]
                        if hist and hist[-1]['date'].strftime('%Y%m%d') == latest_date:
                            metrics = hist[-1].copy()
                            metrics['Symbol'] = symbol
                            past = hist[:-1]
                            dev_b_5 = np.nan
                            dev_s_5 = np.nan
                            dev_b_20 = np.nan
                            dev_s_20 = np.nan
                            if len(past) >= 5:
                                avg_b_5 = pd.Series([p['bought_volume'] for p in past[-5:]]).mean()
                                avg_s_5 = pd.Series([p['sold_volume'] for p in past[-5:]]).mean()
                                dev_b_5 = ((metrics['bought_volume'] - avg_b_5) / avg_b_5 * 100).round(0) if avg_b_5 > 0 else np.nan
                                dev_s_5 = ((metrics['sold_volume'] - avg_s_5) / avg_s_5 * 100).round(0) if avg_s_5 > 0 else np.nan
                            if len(past) >= 20:
                                avg_b_20 = pd.Series([p['bought_volume'] for p in past[-20:]]).mean()
                                avg_s_20 = pd.Series([p['sold_volume'] for p in past[-20:]]).mean()
                                dev_b_20 = ((metrics['bought_volume'] - avg_b_20) / avg_b_20 * 100).round(0) if avg_b_20 > 0 else np.nan
                                dev_s_20 = ((metrics['sold_volume'] - avg_s_20) / avg_s_20 * 100).round(0) if avg_s_20 > 0 else np.nan
                            metrics['dev_b_5'] = dev_b_5
                            metrics['dev_s_5'] = dev_s_5
                            metrics['dev_b_20'] = dev_b_20
                            metrics['dev_s_20'] = dev_s_20
                            metrics_list.append(metrics)
                    theme_df = pd.DataFrame(metrics_list)
                    theme_df = theme_df.sort_values(by=['buy_to_sell_ratio'], ascending=False)
                    total_buy = theme_df['bought_volume'].sum()
                    total_sell = theme_df['sold_volume'].sum()
                    total_volume_sum = theme_df['total_volume'].sum()
                    aggregate_ratio = total_buy / total_sell if total_sell > 0 else float('inf')
                    
                    st.subheader("Summary Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Bought", f"{total_buy:,.0f}")
                    col2.metric("Total Sold", f"{total_sell:,.0f}")
                    col3.metric("Aggregate Buy Ratio", f"{aggregate_ratio:.2f}")
                    
                    display_df = theme_df.copy()
                    display_df['Short %'] = (display_df['bought_volume'] / display_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    display_df['BOT %'] = display_df['Short %']
                    display_df['Signal'] = display_df['buy_to_sell_ratio'].apply(get_signal)
                    display_df['Bought Dev 5d'] = display_df['dev_b_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Sold Dev 5d'] = display_df['dev_s_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Bought Dev 20d'] = display_df['dev_b_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Sold Dev 20d'] = display_df['dev_s_20'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
                    display_df['buy_to_sell_ratio'] = display_df['buy_to_sell_ratio'].round(2)
                    columns = ['Symbol', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d', 'Signal']
                    st.dataframe(display_df[columns].style.applymap(style_signal, subset=['Signal']).applymap(style_dev, subset=['Bought Dev 5d', 'Sold Dev 5d', 'Bought Dev 20d', 'Sold Dev 20d']))
                else:
                    st.write(f"No data available for {selected_theme}.")

if __name__ == "__main__":
    run()
