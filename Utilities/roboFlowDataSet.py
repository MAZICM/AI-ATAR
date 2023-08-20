from roboflow import Roboflow


def roboflow_dataset():
    
    try:
        # Code that might raise an exception
        rbflw_api_key = input("Enter your API_key: ")
        rbflw_workspace = input("Enter your workspace: ")
        rbflw_project = input("Enter your project: ")
        rbflw_download = input("Enter your Download:")
        rf = Roboflow(api_key=rbflw_api_key)
        project = rf.workspace(rbflw_workspace).project(rbflw_project)
        project.version(2).download(rbflw_download)
    except Exception as e:
        # Code to handle other exceptions
        print(f"An error occurred: {e}")
    else:
        # Code to run if no exception occurred
        print("No errors occurred DONE SUCESS")
    finally:
        # Code that will run regardless of whether an exception occurred
        print("ExitProcess")
    # Code continues here
    print("Press Key To Continue with the rest of the program")
