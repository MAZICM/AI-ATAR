
import subprocess


def stream():

    try:
        # Code that might raise an exception
        task1 = input("task :")
        mode1 = input("mode :")
        model1 = input("model :")
        source1 = input("source :")
        device1 = input("device :")
        name1 = input("name :")
        live_stream = [
            "yolo",
            "task=", task1,
            "mode=", mode1,
            "model=", model1,
            "source=", source1,
            "device=", device1,
            "name=", name1,
            "show=", "True",
        ]
        subprocess.run(live_stream)
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

