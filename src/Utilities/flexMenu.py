from src.Utilities import log
import logging

def display_options(options):
    logger = logging.getLogger(__name__)
    #print("\n\t  ======> Enter your choice : ")
    print("")


    while True:
        try:
            print("")
            for index, option in enumerate(options, start=1):
                print(f"\t\t{index}. {option}")
            choice = int(input("\n\t  ======> Enter the number of your choice: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print("")
                log.logger.warning("Invalid choice. Please select a valid option.")
        except ValueError:
            print("")
            log.logger.error("Invalid input. Please enter a number.")



