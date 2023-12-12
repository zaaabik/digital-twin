import json
import os
import re
from datetime import datetime
from glob import glob
from multiprocessing import Pool
from pathlib import Path

from preprocessing.utils.social_reader.BaseSocialReader import BaseSocialReader, Message
from preprocessing.utils.social_reader.TG import laugh_regex
from preprocessing.utils.utils import remove_links
from tqdm import tqdm

from src.utils.logger import get_pylogger

MY_NAME = "я люблю динозавров"

log = get_pylogger(__name__)

official_message_regex = "(\n?(([ა-ჰ])+|You sent an attachment.|\\w+ liked a message)\n?)"
smile_regex = ":3+"
nl_template = "\n"
merge_many_nl_and_spaces = "[\n| ]{2,}"
BOT_ROLE = "bot"
USER_ROLE = "user"


class InstagramBaseSocialReader(BaseSocialReader):
    chats: list[list[dict]]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_path = os.path.join(self.data_path, "zaaabik", "messages", "inbox")

    def is_my_message(self, message: dict) -> bool:
        """
        Is message sent by current TG user
        Args:
            message: telegram message
        """
        return message["sender_name"].encode("latin1").decode("utf8") == MY_NAME

    @staticmethod
    def all_conversation_paths(base_folder: str):
        """
        All chats store in separate file, this function return paths for all users
        Args:
            base_folder: root folder of instagram information
        """
        all_chat_folders = glob(os.path.join(base_folder, "*"))
        return all_chat_folders

    def read_chats(self) -> list[list[dict]]:
        """Read all conversation."""
        all_chat_folder = self.all_conversation_paths(self.data_path)
        log.info(f"All chat count %s", len(all_chat_folder))

        filtered_chat_folders = all_chat_folder
        log.info(f"Filtered chat count %s", len(filtered_chat_folders))

        if self.debug:
            filtered_chat_folders = filtered_chat_folders[::52]

        with Pool(4) as p:
            total_messages = tqdm(
                p.imap(self.get_all_messages_by_user_path, filtered_chat_folders),
                total=len(filtered_chat_folders),
            )
            filtered_chat_folders = list(total_messages)

        return list(filtered_chat_folders)

    @staticmethod
    def skip_message_condition(message):
        """
        Condition for skip message
        Args:
            message:
        """
        cond = ("content" not in message) or ("share" in message)
        if cond:
            return cond
        shared = "shared a story." in message["content"]
        cond = cond or shared
        return cond

    def parse_message(self, messages):
        results = []
        for message in messages:
            if self.skip_message_condition(message):
                continue

            role = self.get_role_from_message(message)
            results.append(
                {
                    "text": message["content"].encode("latin1").decode("utf8"),
                    "role": role,
                    "date": datetime.fromtimestamp(message["timestamp_ms"] / 1000.0).year,
                }
            )
        return results

    def get_messages(self, path):
        with open(path, encoding="utf-8") as fp:
            conversation = json.load(fp)

        messages = self.parse_message(conversation["messages"])
        messages = messages[::-1]

        return messages

    def get_all_messages_by_user_path(self, user_path: str):
        files = glob(str(Path(user_path, "message_*.json")))
        assert len(files) == 1, "Now we can process only one message file per user"
        messages = []
        for file in files:
            messages.extend(self.get_messages(file))
        return messages

    @staticmethod
    def filter_text(user_messages):
        for user_message in user_messages:
            message = remove_links(user_message["text"])
            message = re.sub(laugh_regex, r"\1ахах\4", message, flags=re.M)
            message = re.sub(smile_regex, "", message)
            message = re.sub(official_message_regex, "", message)

            user_message["text"] = message

        user_messages = [
            user_message for user_message in user_messages if len(user_message["text"])
        ]
        return user_messages

    def filter_chats(self, chats: list[list[dict]]) -> list[list[dict]]:
        total_messages = chats
        total_messages_filtered_by_year_filtered_text = [
            self.filter_text(user) for user in total_messages
        ]
        total_messages_filtered_by_year_filtered_text = [
            user for user in total_messages_filtered_by_year_filtered_text if len(user)
        ]
        total_messages_filtered_by_year_filtered_text = [
            self.remove_user_message_by_cond(user)
            for user in total_messages_filtered_by_year_filtered_text
        ]

        return total_messages_filtered_by_year_filtered_text

    def prepare_data(self) -> list[list[Message]]:
        chats = self.read_chats()
        filtered_chats = self.filter_chats(chats)
        filtered_chats_message = [
            [Message(role=message["role"], content=message["text"]) for message in user]
            for user in filtered_chats
        ]
        return filtered_chats_message
