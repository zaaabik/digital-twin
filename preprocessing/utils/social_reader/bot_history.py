import json
import os
from glob import glob

import requests
from preprocessing.utils.social_reader.BaseSocialReader import (
    USER_ROLE,
    BaseSocialReader,
    Message,
)
from preprocessing.utils.utils import write_jsonl


class BotHistoryBaseSocialReader(BaseSocialReader):
    def is_my_message(self, message: dict) -> bool:
        return message["role"] == USER_ROLE

    def prepare_data(self) -> list[list[Message]]:
        files = glob(os.path.join(self.data_path, "*.json"))
        dialogs = []
        for file in files:
            with open(file) as f:
                dialog = json.load(f)
                messages = dialog["messages"]
                messages = [
                    Message(role=message["role"], content=message["context"])
                    for message in messages
                ]
            dialogs.append(messages)
        return dialogs


def download_data(api_path: str, output_file: str):
    """
    Check length of message
    Args:
        api_path: web path to rest service
        output_file: path to save answer of service
    """
    users = requests.get(api_path).json()
    total_output = [{"messages": user["context"]} for user in users]
    write_jsonl(total_output, output_file)
