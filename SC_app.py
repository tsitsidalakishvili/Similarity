import streamlit as st
import pandas as pd
import pandas_profiling  # Import pandas-profiling
import plotly.express as px
import numpy as np
from jira import JIRA
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from similarity import preprocess_data, calculate_similarity
import requests
import subprocess
import pandas as pd
from neo4j import GraphDatabase, basic_auth
import requests
import streamlit as st
import pandas as pd
import streamlit_pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import neo4j
from neo4j_integration import Neo4jManager
import io  # Import the 'io' module for working with byte streams
import csv
nltk.download('stopwords')





def create_neo4j_driver():
    return GraphDatabase.driver(
        "bolt://3.86.204.9:7687",  # Update with your Neo4j hostname and port
        auth=basic_auth("neo4j", "preference-transaction-revision")  # Replace with your Neo4j username and password
    )


def execute_cypher_query(driver, cypher_query, params=None):
    with driver.session() as session:
        result = session.run(cypher_query, params)
        return result.data()



# Set Streamlit configurations
st.set_page_config(layout="wide")  # Use the wide layout

def run():
    if 'session_state' not in st.session_state:
        st.session_state['session_state'] = _SessionState()

    session_state = st.session_state['session_state']

# Define session state
class _SessionState:
    def __init__(self):
        """Initialize session state."""
        self._charts = []
        self._column_mapping = {}  # To store column name changes

    def add_chart(self, chart):
        """Add a chart to the session state."""
        self._charts.append(chart)

    def get_charts(self):
        """Get all charts in the session state."""
        return self._charts

    def get_column_mapping(self):
        """Get the column name mapping."""
        return self._column_mapping

    def set_column_mapping(self, column_mapping):
        """Set the column name mapping."""
        self._column_mapping = column_mapping

if 'session_state' not in st.session_state:
    st.session_state['session_state'] = _SessionState()

session_state = st.session_state['session_state']

st.title("Make Data Talk")




st.sidebar.title("Follow tabs")

tabs = ["Data Source", "Similarity Analysis"]
current_tab = st.sidebar.radio("Select tab", tabs)


tab_descriptions = {
    "Data": "In the Data tab, you can upload your project data or connect to your Jira instance to fetch real-time data."
}

# Create an expander for the tab description
with st.sidebar.expander("Tab Description"):
    if current_tab in tab_descriptions:
        st.write(tab_descriptions[current_tab])




def profile_data_func(df):
    with st.expander("Profile data"):
        pr = df.profile_report()
        st.components.v1.html(pr.to_html(), width=900, height=600, scrolling=True)



def similarity_func(df):
    with st.expander("Similarity Functionality"):
        st.subheader("Similarity Results")

        # Use session state to retrieve columns, remove the 'Save Columns' button
        text_column = st.session_state.get('text_column', df.columns[0])
        identifier_column = st.session_state.get('identifier_column', df.columns[0])
        additional_columns = st.session_state.get('additional_columns', df.columns[0])

        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.05)

        if st.button('Start Similarity Analysis'):
            # Check if columns exist
            if set([st.session_state.text_column, st.session_state.identifier_column] + st.session_state.additional_columns).issubset(set(df.columns)):
                try:
                    # Ensure text_column is of string type
                    df[st.session_state.text_column] = df[st.session_state.text_column].astype(str)
                    
                    # Preprocess and calculate similarity
                    preprocessed_data = preprocess_data(df, st.session_state.text_column)
                    similar_pairs = calculate_similarity(df, threshold, st.session_state.identifier_column, st.session_state.text_column, st.session_state.additional_columns)

                    # Diagnostic outputs
                    st.write(f"Number of rows in the original data: {len(preprocessed_data)}")
                    st.write(f"Number of similar pairs found: {len(similar_pairs)}")

                    # Display similarity results
                    st.subheader(f"Similarity Threshold: {threshold}")
                    st.dataframe(similar_pairs)
                except Exception as e:
                    st.error(f"Error running similarity analysis. Error: {str(e)}")
            else:
                st.error("Selected columns are not present in the data. Please check the column names and try again.")





if current_tab == "Data Source":
    data_source = st.radio("Choose Data Source", ["Upload CSV", "Connect to Jira", "Connect to Neo4j"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
            st.session_state['data_frame'] = df  # This line is similar to the one in the "Connect to Jira" section
            st.write(df)
            st.success("Data successfully uploaded!")

            # UI elements for column selection
            text_column = st.selectbox("Select Text Column for Analysis", df.columns, key='text_column_selector')
            identifier_column = st.selectbox("Select Identifier Column", df.columns, key='identifier_column_selector')
            additional_columns = st.multiselect("Select Additional Columns to Display", df.columns, key='additional_columns_selector')
            
            # Save selected columns
            if st.button('Save Selected Columns for Analysis', key='save_selected_columns_button'):           
                st.session_state.text_column = text_column
                st.session_state.identifier_column = identifier_column
                st.session_state.additional_columns = additional_columns
            # Check if necessary columns are in session_state before calling similarity function
            
            
            if all(key in st.session_state for key in ['text_column', 'identifier_column', 'additional_columns']):
                similarity_func(df)


   

    if data_source == "Connect to Jira":
        jira_url = st.text_input("Jira URL", "https://.atlassian.net")
        jira_email = st.text_input("Email", "tsitsino.dalakishvili@eigen.co")
        jira_token = st.text_input("Jira API Token", "", type="password")
        jql_query = st.text_area("JQL Query", "Sprint = 227 AND assignee in (5ae0553054919b2467109b28) ORDER BY Rank ASC")
        payload = {
            "url": jira_url,
            "email": jira_email,
            "token": jira_token,
            "jql": jql_query
        }

        if st.button("Fetch data using Node-RED"):
            response = requests.post('http://127.0.0.1:1880/fetchJiraData', json=payload)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                    st.session_state['data_frame'] = df 
                    st.write(df)

        # If 'data_frame' exists in the session state, display column selection UI
        if 'data_frame' in st.session_state:
            df = st.session_state['data_frame']
            
            # UI elements for column selection
            text_column = st.selectbox("Select Text Column for Analysis", df.columns, key='text_column_selector')
            identifier_column = st.selectbox("Select Identifier Column", df.columns, key='identifier_column_selector')
            additional_columns = st.multiselect("Select Additional Columns to Display", df.columns, key='additional_columns_selector')
            

            # When the button is clicked
            if st.button('Save Selected Columns for Analysis', key='save_selected_columns_button'):           
                st.session_state.text_column = text_column
                st.session_state.identifier_column = identifier_column
                st.session_state.additional_columns = additional_columns
                
                # Send columns to Node-RED
                columns_payload = {
                    'text_column': text_column,
                    'identifier_column': identifier_column,
                    'additional_columns': additional_columns
                }
                response = requests.post('http://127.0.0.1:1880/saveColumns', json=columns_payload)

                # Handle response from Node-RED
                if response.status_code == 200:
                    st.success("Columns saved for analysis and sent to Node-RED!")
                else:
                    st.error("Failed to send columns to Node-RED.")



            # Check if necessary columns are in session_state before calling similarity function
            if all(key in st.session_state for key in ['text_column', 'identifier_column', 'additional_columns']):
                similarity_func(df)









if __name__ == "__main__":
    run()


# Create a spacer to push content to the bottom of the sidebar
st.sidebar.markdown("---")

# Add the "Created by" line at the bottom of the sidebar
st.sidebar.markdown("Created by [Tsitsi Dalakishvili](https://www.linkedin.com/in/tsitsi-dalakishvili/)")



if 'command' in st.session_state:
    command = st.session_state['command']
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    st.session_state['command_result'] = {
        'returncode': result.returncode,
        'stdout': result.stdout.decode('utf-8'),
        'stderr': result.stderr.decode('utf-8'),
    }

if 'command_result' in st.session_state:
    result = st.session_state['command_result']
    st.subheader("Command Execution Result")
    st.write(f"Return Code: {result['returncode']}")
    st.write(f"Standard Output:\n{result['stdout']}")
    st.write(f"Standard Error:\n{result['stderr']}")
