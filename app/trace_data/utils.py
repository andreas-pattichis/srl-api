from tortoise.contrib.pydantic import pydantic_model_creator
from app.core.config import settings
import pandas as pd

from app.trace_data.models import TraceData


def pydantic_model():
    return pydantic_model_creator(TraceData)


# reading in the label csv to assign each pattern label to a type and subtype
def load_label_meanings():
    # abel names is constant and used to map individual labels to the parent process
    print(settings.PATHS.LABEL_NAMES_CSV)
    labels_df = pd.read_csv(settings.PATHS.LABEL_NAMES_CSV)
    sub_dict = {}
    main_dict = {}
    color_dict = {}

    # reading in the type of each pattern label and creating a dictionary for mapping
    for index, row in labels_df.iterrows():
        sub_dict[row["Pattern No."]] = row["Sub-category"]
        color_dict[row["Pattern No."]] = row["color"]
        m_pattern = row["Pattern No."]
        if m_pattern[0] == "M":
            main_dict[row["Pattern No."]] = "Metacognition"
        else:
            main_dict[row["Pattern No."]] = "Cognition"
        main_dict["NO_PATTERN"] = "NO_PATTERN"

    return sub_dict, main_dict, color_dict


async def model_to_df(trace_data):
    username = list()
    save_time = list()
    # Process start time is in ms
    process_start_time = list()
    process_end_time = list()
    process_label = list()

    n_items = len(trace_data)

    last_process_label = None

    for i, data in enumerate(trace_data):
        data.save_time = int(data.save_time)  # See if we can automatically make this field an int

        if i == 0:  # First element is the essay start time
            essay_start_time = data.save_time

        if i != n_items - 1:
            end_time = int(trace_data[i + 1].save_time) - essay_start_time - 1
        else:
            end_time = int(trace_data[i].save_time) - essay_start_time

        if end_time > settings.MAX_TIME:
            end_time = settings.MAX_TIME

        if last_process_label != data.process_label:
            username.append(data.username)
            save_time.append(data.save_time)
            process_label.append(data.process_label)
            last_process_label = data.process_label

            process_start_time.append(data.save_time - essay_start_time)
            process_end_time.append(end_time)
        else:
            process_end_time[-1] = end_time


    df = pd.DataFrame(data={
        'username': username,
        'process_start_time': process_start_time,
        'process_end_time': process_end_time,
        'process_label': process_label
    })

    return df


async def load_process_features_study(sub_dict, main_dict, color_dict, data):
    # getting the data of the specific student
    # here we read the pattern labels from the flora server

    # milliseconds divided by 45 minutes (in ms)
    data["process_end_time"] = data["process_end_time"] / settings.MAX_TIME
    data["process_start_time"] = data["process_start_time"] / settings.MAX_TIME

    # adding extra columns to the data frame
    data["process_time_spend"] = data["process_end_time"] - data["process_start_time"]
    data["process_sub"] = data["process_label"].map(sub_dict)
    data["process_main"] = data["process_label"].map(main_dict)
    data["color"] = data["process_label"].map(color_dict)

    time_scaler = settings.MAX_TIME / 60000

    # return the full user df
    return data, time_scaler


async def create_series(df, cog_type, time_scaler):
    print("Create_series df")

    # blank colour picker:
    blank_colour = "#ebebeb"

    # selecting the correct type of labels
    if cog_type == 'Combined':
        m_df = df[(df["process_main"] == 'Metacognition') | (df["process_main"] == 'Cognition')]
    else:
        m_df = df[df["process_main"] == cog_type]
    m_df = m_df[
        ["process_start_time", "process_end_time", "process_time_spend", "process_sub", "color"]].reset_index(
        inplace=False)

    m_df = m_df[
        ["process_start_time", "process_end_time", "process_time_spend", "process_sub", "color"]].reset_index(
        drop=True)

    # now we iterate through each row of the df and if there is a gap between two processes we fill the gap with a BLANK
    m_np = []
    for i, row in m_df.iterrows():
        m_row = []
        if (row["process_start_time"] - m_df.iloc[i - 1, 1]) > 0.00001:
            m_np.append(
                [m_df.iloc[i - 1, 1], row["process_start_time"], row["process_start_time"] - m_df.iloc[i - 1, 1],
                 "Niet Gedetecteerd", blank_colour])
        m_np.append(row.to_list())

    # adding a blank at the end in case the last process is of the other type of label
    m_np.append([m_df.iloc[-1, 1], 1, 1 - m_df.iloc[-1, 1], "Niet Gedetecteerd", blank_colour])
    m_df = pd.DataFrame(m_np,
                        columns=["process_start_time", "process_end_time", "process_time_spend", "process_sub",
                                 "color"])
    m_df["process_time_spend"] = m_df["process_time_spend"] * time_scaler
    m_df["process_end_time"] = m_df["process_end_time"] * time_scaler
    m_df["process_start_time"] = m_df["process_start_time"] * time_scaler

    # having created the dataframe we now just have to create the series of data
    series = []
    for i, row in m_df.iterrows():
        if row["process_time_spend"] > 0:
            row_dic = {"name": row["process_sub"], "data": [row["process_time_spend"]], "color": row["color"]}
            series.append(row_dic)

    # the order specified
    orders = {"Metacognition": ["Orientatie", "Plannen", "Monitoren", "Evaluatie"],
              "Cognition": ["Lezen", "Herlezen", "Schrijven", "Verwerking / Organisatie"],
              "Combined": ["Orientatie", "Plannen", "Monitoren", "Evaluatie", "Lezen", "Herlezen", "Schrijven",
                           "Verwerking / Organisatie"]}

    # getting the percentages of each process, along with time until started and time spent on it
    perc = []
    personal = {}
    process_order = list(m_df["process_sub"])

    for i in orders[cog_type]:
        row_dic = {}
        row_dic["name"] = i

        # percentage
        row_dic["data"] = m_df[m_df["process_sub"] == i]["process_time_spend"].sum() / (
            m_df["process_end_time"].max())

        # minutes spent on it
        personal[i + "Mins"] = m_df[m_df["process_sub"] == i]["process_time_spend"].sum()

        # started at minute:
        if i in process_order:
            personal[i + "Start"] = m_df[m_df["process_sub"] == i]["process_start_time"].min()
        else:
            personal[i + "Start"] = 0

        perc.append(row_dic)

    return [series, perc, personal]
