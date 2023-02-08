import email
import os
import re
import nlt
import xgboost as xgb
import streamlit as st
import pandas as pd
import numpy

## Loadin the created model
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

##Caching the model for faster loading
@st.cache
