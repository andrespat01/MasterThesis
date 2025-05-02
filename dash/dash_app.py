import json
from azure.cosmos import CosmosClient
import logging
import pandas as pd
import numpy as np
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosBatchOperationError
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, constraints, regularizers, initializers
import tensorflow as tf
from sklearn.model_selection import ParameterSampler
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer, KNNImputer
from plotly import express as px
from dash import Dash, html, dcc


merged = pd.read_parquet("chargebox_metrics_april.csv")

##### LOCATION STATISTICS #####
# group data by location
location_df = (
    merged_df.groupby("location")
    .agg(
        {
            # locations columns
            "location_latitude": "first",
            "location_longtitude": "first",
            "location_name": "first",
            "location_street_address": "first",
            "location_zipcode": "first",
            "location_city": "first",
            "location_country": "first",
            # metrics columns
            "power_delivered": "sum",
            "sessions": "sum",
            "faulted_sessions": "sum",
            "succesful_sessions": "sum",
            "session_duration": "sum",
            "faults_after_5min": "sum",
            "faulted": "sum",
            "preparing": "sum",
            "charging": "sum",
            "suspendedEVSE": "sum",
            "zero_kwh_sessions": "sum",
            "status_logcount": "sum",
            "response_time": "sum",
            "faults_out_session": "sum",
            "faults_inside_session": "sum",
            "fault_duration": "sum",
            "terminating_fault": "sum",
            "offline_duration": "sum",
            "online_duration": "sum",
            "online_flag_changes": "sum",
            "uptime": "mean",
            "avg_response_time": "mean",
            "charging_succes_rate": "mean",
            "utilization_rate": "mean",
            # other columns
            "chargeBoxId": "nunique",
        }
    )
    .reset_index()
)

# filter to only keep locations with more than 10 chargeboxes
location_df_filtered = location_df[location_df["chargeBoxId"] > 1]
# drop missing locations
location_df_filtered = location_df_filtered.dropna(
    subset=["location_latitude", "location_longtitude"]
)

# drop location with '' in location_latitude and location_longtitude
location_df_filtered = location_df_filtered[
    (location_df_filtered["location_latitude"] != "")
    & (location_df_filtered["location_longtitude"] != "")
]

location_df_filtered["location_latitude"] = location_df_filtered[
    "location_latitude"
].astype(float)
location_df_filtered["location_longtitude"] = location_df_filtered[
    "location_longtitude"
].astype(float)


# 3. Create the mapbox bubble map
fig = px.scatter_map(
    location_df_filtered,
    lat="location_latitude",
    lon="location_longtitude",
    size="chargeBoxId",
    color="charging_succes_rate",
    color_continuous_scale="RdYlGn",
    range_color=(
        location_df_filtered["charging_succes_rate"].min(),
        location_df_filtered["charging_succes_rate"].max(),
    ),
    size_max=15,
    zoom=3,
    hover_name="location_name",
    hover_data={"chargeBoxId": True, "charging_succes_rate": True},
    width=1600,
    height=800,
)

# 4. Tidy up layout
fig.update_layout(
    title="Charging Success Rate & # of Chargeboxes by Location",
    margin={"l": 0, "r": 0, "t": 40, "b": 0},
)

fig.show()


app = Dash(
    __name__,
    requests_pathname_prefix="/dashapp/",
    assets_folder="assets",  # if you have static files
)
server = app.server  # the Flask “app” for Gunicorn

app.layout = html.Div([html.H1("Chargeboxes by Location"), dcc.Graph(figure=fig)])

if __name__ == "__main__":
    app.run(debug=True)  # for local dev
