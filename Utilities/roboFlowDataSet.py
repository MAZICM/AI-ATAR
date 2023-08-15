from roboflow import Roboflow

def roboflow_dataset():
    #rbflw_api_key = "BtcWZsdvGqmTQ2Sfmgbj"
    #rbflw_workspace = "moho-rahimi-xyr0w"
    #rbflw_project = "wildfire-ryisc"
    #rbflw_download = "yolov8"

    rbflw_api_key = input("Enter your API_key: ")
    rbflw_workspace = input("Enter your workspace: ")
    rbflw_project = input("Enter your project: ")
    rbflw_download= input("Enter your Download:")
    rf = Roboflow(api_key=rbflw_api_key)
    project = rf.workspace(rbflw_workspace).project(rbflw_project)
    dataset = project.version(2).download(rbflw_download)
