import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import pandas as pd
from yahoofinancials import YahooFinancials
startDate = '2004-04-30'
endDate = '2023-12-31'
freq = 'monthly'

tickers = ['^GSPC','^IBEX','^STOXX','^IXIC','^N225','^FTSE']
yahoo_financials = YahooFinancials(tickers)
data = yahoo_financials.get_historical_price_data(start_date=startDate, 
                                                  end_date=endDate, 
                                                  time_interval=freq)

aux = pd.DataFrame(data['^STOXX']['prices'])
aux.head()




