"""Mridul Mridul, 40279215
Reema Aboudraz, 40253549
COMP-472 Summer 2026
Mini-Project 2 Submission: Spam/ Ham Email and Message Classifier"""

from spam_detector import SpamDetector

DATASET_PATH = "spam.csv"


def print_welcome_banner():
    banner = r"""
    🌸🐱🌸🐱🌸🐱🌸🐱🌸🐱🌸🐱🌸
    
        Welcome to Spam Detection AI
                 made by
        Mridul Mridul & Reema Aboudraz

    Type a message and I'll tell you if it's spam!

    Type 'quit' anytime to exit.
    
    🌸🐱🌸🐱🌸🐱🌸🐱🌸🐱🌸🐱🌸
"""
    print(banner)


def main():
    print_welcome_banner()

    detector = SpamDetector(DATASET_PATH)

    try:
        detector.load_data()
    except FileNotFoundError as error:
        print(error)
        return 

    detector.prepare_features()
    print("\nTraining model...")
    detector.train()

    detector.evaluate()

    # DATA VISUALIZATION (bar chart) -- to be implemented -- can use SpamDetector.plot_distribution() function here

    while True:
        try:
            message = input("\nEnter message:\n> ").strip()
        except EOFError:
            break

        if message.lower() == "quit":
            break

        if message == "":
            print("Please type something.")
            continue

        label = detector.predict_message(message)
        print(f"Prediction: {label.upper()}")

        # CONFIDENCE SCORE -- to be implemented -- can use SpamDetector.predict_message() function here

    print("Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!🌸")
