import base64
import json
import os
import warnings

import requests
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.llms.azure_openai import AzureOpenAI
from Tools import execute_command, ask_user_for_input, execute_python_file, get_date_time_and_weeknumber, upload_and_ask_image, create_folder, get_upload_and_ask_image_as_text
from pathlib import Path

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# region Tools
ExecTool = FunctionTool.from_defaults(execute_command, name="execute_command",
                                      description="Execute a command on the local Windows system. Use only Windows commands. You can do file and folder reading, manipulation, run programs, and more.")
AskUserTool = FunctionTool.from_defaults(ask_user_for_input, name="ask_user_for_input",
                                         description="Ask the user for input.")
ExecutePythonTool = FunctionTool.from_defaults(execute_python_file, name="execute_python_file",
                                               description="Execute Python code stored in .py file.")
DateDayTimeWeekTool = FunctionTool.from_defaults(get_date_time_and_weeknumber, name="get_date_time_and_weeknumber",
                                                 description="Get the current date, day, time and week number.")
# UploadAndAskImageTool = FunctionTool.from_defaults(upload_and_ask_image, name="upload_and_ask_image",
#                                                    description="Upload an image to the LLM, ask a question about it, and get the answer.")
GetUploadedImageToolAsText = FunctionTool.from_defaults(get_upload_and_ask_image_as_text, name="get_upload_and_ask_image_as_text", description="Get the code for the upload_and_ask_image function as plain text.")
CreateFolderTool = FunctionTool.from_defaults(create_folder, name="create_folder", description="Create a new folder by providing the full path of the folder to be created.")

tools = [ExecTool, AskUserTool, ExecutePythonTool, DateDayTimeWeekTool, GetUploadedImageToolAsText, CreateFolderTool]
# endregion

llm = AzureOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)


class ExpertReActAgent:

    def __init__(self):
        global tools, llm
        self.project_folder = None
        self.agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True, max_iterations=100)
        self.START_PROJECT_PROMPT = """
        You are an expert Python programming assistant. Your task is to create a new Python project based on the user's description.
        Ask the user for a detailed description of the project goals, then create all necessary code to accomplish these goals. If you are unsure how to create or complete the python code, decide whether to search the internet for the required information or consult the existing code files.
        """

        self.CODE_PROMPT = """
        Given the existing code you have read so far in the files of the current working directory and the user's project description, decide what additional code needs to be written. If you think you are done, then only output the word 'DONE' without any other words. If not done or existing code can be optimized, then create that new code. Write the new code and ensure it fits well with the existing code, meaning: if the existing code is extended then write the entire new code to that file, otherwise create a new code file.
        When writing the code to a new file instead of in the extending existing code file, give the file a meaningful name that reflects its purpose. Use .py as extension.
        """

    def read_existing_code(self):
        """
        Read existing code files in the project directory.
        :return: A dictionary containing the existing code files.
        """
        existing_code = {}
        current_dir = self.project_folder  # Get the current working directory as a Path object
        for file in current_dir.iterdir():  # Iterate through the directory contents
            if file.is_file() and file.suffix == ".py":  # Check if the file is a Python file
                with open(file, "r") as f:
                    existing_code[file.name] = f.read()  # Read and store the file's contents
        if existing_code == {}:
            return "No existing code files found. Retry creating the codefiles for the project"
        return existing_code

    def write_to_file(self, txt: str, file_name: str):
        """
        Write text to a new file.
        :param txt: The text to be written to the file.
        :param file_name: The name of the file to be created.
        """
        current_dir = self.project_folder
        file_path = current_dir / file_name
        with open(file_path, "w") as f:
            f.write(str(txt))
        print(f"Code written to file: {file_path}")
        return f"Code written to file: {file_path}"

    def set_variable_project_folder(self, project_folder: str):
        """
        Set the project folder for the agent.
        :param project_folder: The path to the project folder.
        """
        self.project_folder = Path(project_folder)
        return f"Project folder set to: {self.project_folder}"

    def chat_with_agent(self):
        """
        Chat with the agent and get the response.
        :return: The response from the agent.
        """
        conversation_history = [ChatMessage(role=MessageRole.SYSTEM, content=self.START_PROJECT_PROMPT)]
        blnInit = True
        user_input = ""
        WriteToFileTool = FunctionTool.from_defaults(self.write_to_file, name="write_to_file",
                                                         description="Write text to a file with a specific name. Isf the file exists, it will be overwritten")
        SetVariableProjectFolderTool = FunctionTool.from_defaults(self.set_variable_project_folder, name="set_variable_project_folder",
                                                                  description="Set the project_folder variable for the agent.")
        tools.append(WriteToFileTool)
        tools.append(SetVariableProjectFolderTool)

        while True:
            if blnInit:
                user_input = input("Please provide a detailed description of the goals for the Python code to be written: ")
                user_input = f"Description: {user_input}" + "\n\n" + """As a first step, create a new folder for the project. Give it a meaningful name, derived from the given description. 
                Set variable project_folder to the new name. Change the current working directory to that new folder and write the initial code into a Python (.py) file or multiple Python files that must be placed into the new folder. Make sure that the imports also contain 'from dotenv import load_dotenv', and the load_dotenv() function is called.
                Create a .env file in the new folder and add the following environment variables when applicable (for each single variable): OPENAI_MODEL, OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION.
                Give the file or files a meaningful name or names that reflect the purpose of the code within. Make sure the code in the files have Docstrings in reStructuredText (reST) format. 
                Create a requirements.txt with the python libraries that need to be installed.
                Don't use placeholders in the code, but write the complete- code instead. When you are done, create an extensive README.md file that contains a full elaborated user guide in the new folder. Make sure the README.md file contains the following sections: Introduction, Installation (including setting the .env variables), Usage, Contributing, License, and Acknowledgements. 
                IMPORTANT: When there is need for OCR, it is MANDATORY to use the UploadAndAskImageTool from the Tools.py module!! Read the code from this Tools.py file as raw plain text (even for json) and copy the UploadAndAskImageTool function as a function into the new code file."""
                # Ensure user_input is a string
                if not isinstance(user_input, str):
                    user_input = str(user_input)
                cm = ChatMessage(role=MessageRole.USER, content=user_input)
                response = self.agent.chat(message=user_input, chat_history=conversation_history)
                conversation_history.append(cm)
                print("Assistent: ", response.response)
                cm = ChatMessage(role=MessageRole.ASSISTANT, content=response.response)
                conversation_history.append(cm)
                blnInit = False
            else:
                code = self.read_existing_code()
                # Ask the agent if it needs to create new code or extend existing code
                prompt = self.CODE_PROMPT + f"\n\nExisting code files: {json.dumps(code)}"
                cm = ChatMessage(role=MessageRole.USER, content=prompt)
                response = self.agent.chat(message=prompt, chat_history=conversation_history)
                conversation_history.append(cm)
                print("Assistent: ", response.response)
                cm = ChatMessage(role=MessageRole.ASSISTANT, content=response.response)
                conversation_history.append(cm)
                if response.response == "DONE":
                    last_check = f"Perform a final check of the code files in the project folder: {self.project_folder} and make sure all functions can be correctly called and executed. If everything is correct, then the project is completed. Return only the word 'DONE' without any other words when you are done."
                    cm = ChatMessage(role=MessageRole.USER, content=last_check)
                    conversation_history.append(cm)
                    response = self.agent.chat(message=last_check, chat_history=conversation_history)
                    if response.response == "DONE":
                        print("Project completed.")
                        break


if __name__ == "__main__":
    agent = ExpertReActAgent()
    agent.chat_with_agent()
