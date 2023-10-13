from src.Utilities import features
import logging
from src.Utilities import log
import time


def display_menu():
    # Create a custom logger
    # logger.error("An error occurred: %s", e, exc_info=True)
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    start_time = time.time()
    # Code to measure performance
    # Example usage
    # log.logger.debug("This is a debug message.")
    # log.logger.info("This is an info message.")
    # log.logger.warning("This is a warning message.")
    # log.logger.error("This is an error message.")
    # log.logger.critical("This is a critical message.")
    log.logger.info("Welcome To ATAR ! :)")

    print("\n\n\t-----------------------")
    print("\tWelcome to My CLI Menu")
    print("\t-----------------------\n")
    print("\t\t1. Download RoboFlow straining dataset")
    print("\t\t2. Train")
    print("\t\t3. Resume existing Train")
    print("\t\t4. Valid")
    print("\t\t5. Live Test")
    print("\t\t6. test on an existing file")
    print("\t\t7. Quit")
    print("\n\t----------------------------------------------------------")
    print("\tTo exit the CLI menu, choose option '7' or press 'Ctrl+C'.")
    print("\t------------------------------------------------------------\n")
    end_time = time.time()
    log.logger.info("Execution time: %.2f seconds", end_time - start_time)
    print("\n\n")


def get_choice():
    choice = input("\n\t  ======> Enter your choice : ")
    return choice


def get_resp(choice):
    if choice == '1':
        print("Downloading DataSet ...............................")
        features.get_dataset()
        return 1
    elif choice == '2':
        print("\n\t DEFAULT MODELS :")
        features.train()
        return 1
    elif choice == '4':
        print("\n\t Trained Model to Validate : ")
        features.valid()
        return 1
    elif choice == '5':
        print("Stream..............")
        features.stream()
        return 1
    elif choice == '6':
        print("V..............")
        features.video_detect()
        return 1
    elif choice == '3':
        print("RT..............")
        features.resume_train()
        return 1
    elif choice == '7':
        log.logger.info("\nThank you for Running me ! \nGood bye ! :)\n")
        return 0

    else:
        log.logger.warning("\nInvalid choice. Please select a valid option !!!!!!!!!!")


def main():
    x = 1
    while x != 0:
        print("\n")
        display_menu()
        choice = get_choice()
        x = get_resp(choice)


if __name__ == "__main__":
    main()
