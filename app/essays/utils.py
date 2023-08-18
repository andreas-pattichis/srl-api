from tortoise.contrib.pydantic import pydantic_model_creator
import pandas as pd

from app.core.config import settings

from app.essays.models import Essay


def pydantic_model():
    return pydantic_model_creator(Essay)


async def validate_user_exists(username):
    return await pydantic_model().from_queryset_single(Essay.get(username=username))


# reading in the label csv to assign each pattern label to a type and subtype
def load_label_meanings():
    # label names is constant and used to map individual labels to the parent process
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


def load_process_features_study_f(sub_dict, main_dict, color_dict, f):
    # getting the data of the specific student
    # here we read the pattern labels from the flora server
    data = pd.read_csv(settings.PATHS.PROCESS_LABEL_DIR + f)
    data = data[data["Process End Time"] > -1]

    data["Process End Time"] = data["Process End Time"] / settings.MAX_TIME
    data["Process Start Time"] = data["Process Start Time"] / settings.MAX_TIME

    # adding extra columns to the data frame
    data["Process_Time_Spent"] = data["Process End Time"] - data["Process Start Time"]
    data["Process_sub"] = data["Process Label"].map(sub_dict)
    data["Process_main"] = data["Process Label"].map(main_dict)
    data["Color"] = data["Process Label"].map(color_dict)

    time_scaler = settings.MAX_TIME / 60000

    # return the full user df
    return data, time_scaler
