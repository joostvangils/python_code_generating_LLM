import base64
import os
import subprocess
import time
from io import BytesIO
from typing import Tuple

import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
PROJECT_FOLDER = os.getenv("PROJECT_FOLDER")


def get_date_time_and_weeknumber():
    """
    Get the current date, day, time and week number.
    :return: The current date, day, time and week number.
    """
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%A"), now.strftime("%U")


def execute_command(command: str = ''):
    """
    Execute a command-line command and return the output.
    :param command: The command to execute.
    :return: The output of the command.
    """
    try:
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        # Check if there were any errors
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        # Return the output of the command
        return result.stdout.strip()
    except Exception as e:
        return f"Exception: {str(e)}"


def execute_python_file(file_path: str = ''):
    """
    Execute a Python file and return the output.
    :param file_path: The path to the Python file.
    :return: The output of the file.
    """
    try:
        # Execute the Python file
        result = subprocess.run(['python', file_path], capture_output=True, text=True)
        # Check if there were any errors
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        # Return the output of the file
        return result.stdout.strip()
    except Exception as e:
        return f"Exception: {str(e)}"


def ask_user_for_input(ask: str = ''):
    """
    Ask the user for input.
    :param ask: The input should specify the question to ask the user.
    :return: The user's input.
    """
    inputs = str(ask + " ")
    return input(inputs)


def upload_and_ask_image(image_path: str, question: str) -> str:
    """
    Uploads an image to the LLM, asks a question about it, and returns the answer.
    :param image_path: Path to the image file
    :param question: The question to ask about the image
    :return: The answer from the LLM
    """
    # Configuration
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    IMAGE_PATH = image_path
    encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    # Payload for the request
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that helps people find information."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            },

        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }
    ENDPOINT = f"{os.getenv('OPENAI_API_BASE')}/openai/deployments/{os.getenv('OPENAI_DEPLOYMENT_NAME')}/chat/completions?api-version={os.getenv('OPENAI_API_VERSION')}"
    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload, verify=False)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Handle the response as needed (e.g., print or process)
    return response.json()["choices"][0]["message"]["content"]


def create_folder(full_folder_path: str):
    """
    Create a folder if it does not exist.
    :param full_folder_path: The full path of the folder to create.
    :return: The full path of the folder.
    """
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)
    return full_folder_path


def get_upload_and_ask_image_as_text():
    """
    Get the text of the function upload_and_ask_image.
    :return: The text of the function upload_and_ask_image.
    """
    return upload_and_ask_image.__doc__
