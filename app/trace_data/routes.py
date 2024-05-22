import os
import pickle

import joblib
import numpy as np
from fastapi import APIRouter, Response
from tortoise.expressions import Q
from tortoise import connections

from .utils import *

from app.trace_data.models import TraceData, MdlUser, MdlCourse

router = APIRouter(prefix="/api/tracedata", tags=["tracedata"])


def load_model_and_scaler():
    base_path = os.getcwd()  # Gets the current working directory
    model_path = os.path.join('trace_data', 'gmm_model', 'gmm_model.pkl')
    scaler_path = os.path.join(base_path, 'trace_data', 'gmm_model', 'scaler.pkl')

    # Print current working directory of the machine
    print(f"Current working directory: {os.path.abspath(os.getcwd())}")

    # Check if the model and scaler files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found in {scaler_path}")

    # Use joblib to load the model and scaler
    gmm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return gmm_model, scaler


# Load the model and scaler outside the endpoint to avoid reloading on each request
gmm_model, scaler = load_model_and_scaler()

# Features used for scaling
FEATURE_COLUMNS = [
    'Total Cycles',
    'Total Metacognition Actions',
    'Total Cognition Time',
    'Initial Orientation Time',
    'Total High Cognition Actions',
    'Total High Cognition Time',
    'Ratio of High Cognition Time',
    'Total Metacognition Time'
]


@router.get(
    "/{username}",
    status_code=200,
)
async def tracedata_results_from_user(username: str, response: Response):
    db_moodle = connections.get('moodle')

    user = await MdlUser.get(username=username, using_db=db_moodle)
    if not user:
        response.status_code = 404
        return {
            'message': 'User not found'
        }

    course_ids = await TraceData.filter(user_id=user.id, process_label__isnull=False).distinct().values('course_id')

    results = []

    for course_id in course_ids:
        trace_data = await TraceData.filter(user_id=user.id, process_label__isnull=False,
                                            course_id=course_id['course_id']).order_by('save_time')
        trace_data = await model_to_df(trace_data)
        # columns: process_start_time,process_end_time,process_label,process_time_spend
        df = await map_process_labels(trace_data)
        # columns: process_start_time,process_end_time,process_label,process_time_spend, process_sub, process_main, color

        # Copy df to new variable to avoid changing the original df
        df_copy = df.copy()
        df_copy['Username'] = username
        cluster_names, cluster_probs = clustering_results(df_copy)

        # making the data series and percentages for meta and cog
        m_series, m_percentages = await create_series(df, "Metacognition")
        c_series, c_percentages = await create_series(df, "Cognition")
        series, percentages = await create_series(df, "Combined")

        course = await MdlCourse.get(id=course_id['course_id'], using_db=db_moodle)
        name_nl = "Essay " + str(course_id['course_id'])
        name_en = "Essay " + str(course_id['course_id'])
        if course:
            if '{mlang}' in course.fullname:
                name_en = course.fullname.split('{mlang en}')[1].split('{mlang}')[0]
                name_nl = course.fullname.split('{mlang nl}')[1].split('{mlang}')[
                    0] if '{mlang nl}' in course.fullname else name_en
            else:
                name_nl = course.fullname
                name_en = course.fullname
        print(cluster_names)
        result = {
            'course_id': course_id['course_id'],
            'name_nl': name_nl,
            'name_en': name_en,
            'meta': m_series,
            'm_perc': m_percentages,
            'cog': c_series,
            'c_perc': c_percentages,
            'combined_perc': percentages,
            'combined_series': series,

            'cluster_names': cluster_names,
            'cluster_probs': cluster_probs
        }

        results.append(result)

    return results


def preprocess_data(df):
    """
    Preprocesses the provided DataFrame by applying various data manipulation operations such as label replacements,
    new column creation for process duration, time unit conversions, row filtering based on conditions,
    and merging consecutive rows based on criteria.
    """

    # 1. REPLACE LABELS
    # Define the labels to be replaced and their corresponding new labels
    replacement_mapping = {
        'MCO1': 'Orientation',
        'MCO2': 'Orientation',
        'MCO3': 'Orientation',
        'MCO4': 'Orientation',
        'MCO5': 'Orientation'
    }
    # Apply the replacement mapping to the 'Process Label' column
    df['Process Label'] = df['process_label'].replace(replacement_mapping)

    # Rename 'Process End Time' and 'Process Start Time' columns to 'process_end_time' and 'process_start_time'
    df.rename(columns={'process_end_time': 'Process End Time', 'process_start_time': 'Process Start Time'},
              inplace=True)

    # 2. ADD NEW COLUMN FOR PROCESS DURATION
    # Calculate 'Process Duration' by subtracting 'Process Start Time' from 'Process End Time'
    df['Process Duration'] = df['Process End Time'] - df['Process Start Time']

    # 3. CONVERT TIME COLUMNS TO SECONDS
    # Identify columns representing time in milliseconds to convert to seconds
    correct_time_columns = ['Process Start Time', 'Process End Time', 'Process Duration']
    # Convert milliseconds to seconds for the specified columns
    for column in correct_time_columns:
        df[column] = df[column] / 1000.0

    # 4. REMOVE ROWS WITH DURATION LESS THAN 1 SECOND
    # Filter out rows where the process duration is less than one second
    df = df[df['Process Duration'] >= 1]

    # 5. MERGE CONSECUTIVE ROWS WITH EXACT SAME PROCESS LABEL THAT START WITH 'HC.'
    # Initialize a new DataFrame to hold the processed data
    processed_data = []
    # Use a while loop to iterate through the DataFrame because we need to skip an unknown number of rows
    i = 0
    while i < len(df):
        current_row = df.iloc[i].copy()
        # Check if the row's 'Process Label' starts with 'HC.'
        if current_row['Process Label'].startswith('HC'):
            # Initialize variables to accumulate the total duration and track the end time of the last consecutive row
            total_duration = current_row['Process Duration']
            end_time = current_row['Process End Time']
            j = i + 1
            # Loop through following rows to find consecutive rows with the same 'Process Label' and 'Username'
            while j < len(df):
                next_row = df.iloc[j]
                if (current_row['Process Label'] == next_row['Process Label'] and
                        current_row['Username'] == next_row['Username']):
                    total_duration += next_row['Process Duration']
                    end_time = next_row['Process End Time']
                    j += 1
                else:
                    break
            # Update the current row with the accumulated duration and the last end time
            current_row['Process Duration'] = total_duration
            current_row['Process End Time'] = end_time
            # Append the updated row to the processed data list
            processed_data.append(current_row)
            # Skip to the row after the last merged row
            i = j
        else:
            # Append the current row as is to the processed data list if it doesn't start with 'HC.'
            processed_data.append(current_row)
            i += 1

    # Convert the processed data list back into a DataFrame
    df_preprocessed_data = pd.DataFrame(processed_data)

    return df_preprocessed_data


def extract_features(df_preprocessed_data):
    """
    Extracts multiple features from the preprocessed data, focusing on different aspects of user activity
    related to self-regulated learning cycles, high cognition activities, and metacognition measures.

    Features extracted:
    1. Total Cycles
    2. Initial Orientation Time
    3. Total High Cognition Actions
    4. Total High Cognition Time
    5. Total Cognition Time
    6. Ratio of High Cognition Time
    7. Total Metacognition Time
    8. Total Metacognition Actions
    """

    # Feature 1, 2: Number of Self-Regulated Learning (SRL) Cycles, Initial Orientation Time
    user_features = {}
    for row in df_preprocessed_data.itertuples(index=False):
        username = row.Username
        process_label = row[8]
        process_duration = row[9]

        if username not in user_features:
            user_features[username] = {
                'orientation_found': False,
                'cognition_found': False,
                'evaluation_found': False,
                'cycles': 0,
                'initial_orientation_time': 0
            }

        user_data = user_features[username]

        if process_label == "Orientation" or process_label.startswith("MCP"):
            user_data['orientation_found'] = True
            if user_data['cycles'] == 0:
                user_data['initial_orientation_time'] += process_duration

        elif user_data['orientation_found'] and (
                process_label.startswith("HC.EO.") or process_label.startswith("LCF") or process_label.startswith(
            "LCR")):
            user_data['cognition_found'] = True

        elif user_data['cognition_found'] and (process_label.startswith("MCE") or process_label.startswith("MCM")):
            user_data['evaluation_found'] = True

        if user_data['orientation_found'] and user_data['cognition_found'] and user_data['evaluation_found']:
            user_data['cycles'] += 1
            user_data['orientation_found'] = False
            user_data['cognition_found'] = False
            user_data['evaluation_found'] = False

    df_features = pd.DataFrame([
        {
            'Username': username,
            'Total Cycles': data['cycles'],
            'Initial Orientation Time': data['initial_orientation_time']
        }
        for username, data in user_features.items()
    ])

    # Feature 3: Total High Cognition Actions
    for username in user_features:
        user_features[username]['high_cognition_count'] = 0

    for row in df_preprocessed_data.itertuples(index=False):
        username = row.Username
        process_label = row[8]
        if process_label.startswith("HCEO"):
            user_features[username]['high_cognition_count'] += 1

    df_features['Total High Cognition Actions'] = df_features['Username'].apply(
        lambda x: user_features[x]['high_cognition_count'])

    # Features 4, 5, 6: Total High Cognition Time, Total Cognition Time, and Ratio of High Cognition Time
    for username in user_features:
        user_features[username]['high_cognition_time'] = 0
        user_features[username]['total_cognition_time'] = 0

    for row in df_preprocessed_data.itertuples(index=False):
        username = row.Username
        process_label = row[8]
        process_duration = row[9]

        if process_label.startswith("HCEO"):
            user_features[username]['high_cognition_time'] += process_duration
            user_features[username]['total_cognition_time'] += process_duration

        elif process_label.startswith("LCF") or process_label.startswith("LCR"):
            user_features[username]['total_cognition_time'] += process_duration

    df_features['Total High Cognition Time'] = df_features['Username'].apply(
        lambda x: user_features[x]['high_cognition_time'])
    df_features['Total Cognition Time'] = df_features['Username'].apply(
        lambda x: user_features[x]['total_cognition_time'])
    df_features['Ratio of High Cognition Time'] = df_features.apply(
        lambda x: x['Total High Cognition Time'] / x['Total Cognition Time'] if x['Total Cognition Time'] > 0 else 0,
        axis=1)

    # Feature 7: Total Metacognition Time
    for username in user_features:
        user_features[username]['metacognition_time'] = 0

    for row in df_preprocessed_data.itertuples(index=False):
        username = row.Username
        process_label = row[8]
        process_duration = row[9]

        if process_label.startswith("MCP") or process_label.startswith("MCM") or process_label.startswith(
                "MCE") or process_label == "Orientation":
            user_features[username]['metacognition_time'] += process_duration

    df_features['Total Metacognition Time'] = df_features['Username'].apply(
        lambda x: user_features[x]['metacognition_time'])

    # Feature 8: Total Metacognition Actions
    for username in user_features:
        user_features[username]['metacognition_count'] = 0

    for row in df_preprocessed_data.itertuples(index=False):
        username = row.Username
        process_label = row[8]

        if process_label.startswith("MCP") or process_label.startswith("MCM") or process_label.startswith(
                "MCE") or process_label == "Orientation":
            user_features[username]['metacognition_count'] += 1

    df_features['Total Metacognition Actions'] = df_features['Username'].apply(
        lambda x: user_features[x]['metacognition_count'])

    return df_features


def clustering_results(df_copy):
    # Preprocess Data
    df_preprocessed_data = preprocess_data(df_copy)
    # columns: process_start_time,process_end_time,process_label,process_time_spend, process_sub, process_main,
    # color, Username, Process Duration

    # Extract Features
    df_features = extract_features(df_preprocessed_data)
    # columns: Username, Total Cycles, Initial Orientation Time, Total High Cognition Actions, Total High Cognition
    # Time, Total Cognition Time, Ratio of High Cognition Time, Total Metacognition Time, Total Metacognition Actions

    # Filter records with conditions
    df_features_filtered = df_features[
        (df_features['Total Cognition Time'] != 9400.913) & (df_features['Total Cognition Time'] > 600)]

    # Remove the 'Username' column before normalizing
    df_features_filtered = df_features_filtered.drop(columns='Username')

    # Remove outliers
    z_scores = np.abs((df_features_filtered - df_features_filtered.mean()) / df_features_filtered.std())
    df_features_filtered_copy = df_features_filtered[(z_scores < 4.5).all(axis=1)]

    # FIXME: Sometimes the filtered data is empty, so we keep the original data (temporary solution)
    if not df_features_filtered_copy.empty:
        df_features_filtered = df_features_filtered_copy

    # Normalize features and select relevant ones for clustering
    df_filtered_norm = scaler.transform(df_features_filtered)
    # Keep only columns with indices 0, 7, and 4
    df_filtered_norm = df_filtered_norm[:, [0, 7, 4]]

    cluster_names = {
        0: 'confidentProducer',
        1: 'reflectiveWriter',
        2: 'thoughtfulPlanner',
        3: 'efficientScribbler'
    }
    # Cluster the normalized features
    probabilities = gmm_model.predict_proba(df_filtered_norm)
    # Get top 2 probabilities and their corresponding cluster labels for each sample
    top2_indices = np.argsort(probabilities, axis=1)[:, -2:]
    top2_labels = np.array([[top2_indices[i, -1], top2_indices[i, -2]] for i in range(len(top2_indices))])
    top2_probs = [[probabilities[i, top2_indices[i, -1]], probabilities[i, top2_indices[i, -2]]] for i in
                  range(len(probabilities))][0]
    top2_probs = [round(prob, 2) for prob in top2_probs] # Round to 2 digits
    # Get the names of the top 2 clusters
    top2_names = [[cluster_names[label] for label in labels] for labels in top2_labels][0]

    return top2_names, top2_probs
