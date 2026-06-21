from datetime import datetime
import os


def print_file_creation_time(file_name):
    if os.path.exists(file_name):
        creation_time = os.path.getctime(file_name)
        formatted_time = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"The file '{file_name}' was created on: {formatted_time}")
    else:
        print(f"File '{file_name}' does not exist.")
