"""Hard-coded large-cap US equity universe (reused from HCCT Group's Equity-Factor-Model).

The first 100 of 113 liquid S&P 500 constituents (as of 2026) that were already listed
before 2010. NOTE: picking today's survivors introduces survivorship bias — see README.
"""

TICKERS: list[str] = """
AAPL MSFT AMZN GOOGL NVDA BRK-B JPM JNJ V PG UNH HD MA XOM CVX PFE KO PEP MRK ABT
WMT BAC CSCO ORCL INTC VZ T DIS MCD NKE ADBE CRM NFLX QCOM TXN AVGO AMGN GILD BMY LLY
MDT TMO DHR HON UNP UPS CAT DE MMM GE BA LMT RTX IBM ACN WFC C GS MS AXP
BLK SCHW USB PNC COF MET AIG CB SPGI CME ICE COST LOW TGT SBUX TJX BKNG ROST MO PM
CL KMB GIS MDLZ COP EOG SLB OXY NEE DUK SO D AEP EXC AMT PLD SPG CCI ECL SHW
""".split()

MARKET = "SPY"
VIX = "^VIX"
