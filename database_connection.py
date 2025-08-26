import sqlalchemy
from google.cloud.sql.connector import Connector
import streamlit as st
import base64
import json
import os

service_account_info = json.loads(base64.b64decode(st.secrets["service_account_base64"]))

credentials_path = "/tmp/service_account.json"

with open(credentials_path, "w") as f:
    json.dump(service_account_info, f)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path


db_user = st.secrets["db_user"]
db_pass = st.secrets["db_pass"]
db_name = st.secrets["db_name"]
connection_name = st.secrets["instance_connection_name"]



connector = Connector()

def getconn():
    conn = connector.connect(
        connection_name,
        "pymysql",
        user=db_user,
        password=db_pass,
        db=db_name
    )
    return conn


engine = sqlalchemy.create_engine(
    "mysql+pymysql://",
    creator=getconn,
)
