import subprocess


def m_valid():


    try:
        task2 = input("task :")
        mode2 = input("mode :")
        model2 = input("model :")
        data2 = input("data :")
        name2 = input("name :")

        validation = [
            "yolo",
            "task=", task2,
            "mode=", mode2,
            "model=", model2,
            "name=", name2,
            "data=", data2
        ]
        subprocess.run(validation)
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
