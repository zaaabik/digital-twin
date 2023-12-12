import json
import os
import re
from collections import OrderedDict
from copy import deepcopy
from typing import Iterable

from preprocessing.utils.social_reader.BaseSocialReader import BaseSocialReader, Message
from preprocessing.utils.utils import remove_links
from tqdm import tqdm

from src.utils.logger import get_pylogger

MY_ID = "user255314293"
FIELDS_TO_KEEP = ["text", "from_name"]

BOT_ROLE = "bot"
USER_ROLE = "user"
merge_many_nl_and_spaces = "[\n| ]{2,}"
laugh_regex = "(^|[^а-яА-Я])((а|А|в|В|х|Х|X|x|A|aПха|пха|ПХА){4,})($|[^а-яА-Я])"


log = get_pylogger(__name__)


class TGBaseSocialReader(BaseSocialReader):
    chats: list[list[Message]]

    def __init__(self, allowed_chat_types: list, **kwargs):
        """
        Read telegram data
        Args:
            allowed_chat_types: list of chat types processed
        """
        super().__init__(**kwargs)
        self.data_path = os.path.join(self.data_path, "result.json")
        self.allowed_chat_types = allowed_chat_types

    def read_chats(self) -> list[dict]:
        """Read all chats."""
        with open(self.data_path) as f:
            result = json.load(f)

        chats = result["chats"]["list"]
        log.info("Initial chat count %s", len(chats))

        return list(chats)

    def filter_users_cond(self, user_chat) -> bool:
        """
        Filter users
        Args:
            user_chat: user chat
        """
        return user_chat["type"] in self.allowed_chat_types

    @staticmethod
    def filter_messages(messages: list[dict]) -> list[dict]:
        """
        Remove some messages
        Args:
            messages: user messages
        """
        filtered_messages = filter(lambda x: x["type"] == "message" and x["text"], messages)
        return list(filtered_messages)

    @staticmethod
    def merge_any_while_spaces(text: str) -> str:
        """
        Merge multiple new line symbols and spaces
        Args:
            text: message text
        """
        any_whitespace_regexp = "[\t ]+"
        text = re.sub(any_whitespace_regexp, " ", text)
        any_new_line_regexp = "\n+"
        text = re.sub(any_new_line_regexp, "\n", text)
        return text

    @staticmethod
    def process_text(message_body) -> str:
        """
        Some telegram`s message consist of parts not only with text, but with other type (link, images)
        Take only text part
        Args:
            message_body: whole message information
        """
        if isinstance(message_body, str):
            return message_body
        elif isinstance(message_body, Iterable):
            res = ""
            for part in message_body:
                if isinstance(part, dict):
                    res += part["text"]
                else:
                    res += part
            return res
        else:
            raise Exception("Wrong data type passed")

    def get_role_from_message(self, message: dict) -> str:
        if self.is_my_message(message):
            role = BOT_ROLE
        else:
            role = USER_ROLE
        return role

    def is_my_message(self, message: dict) -> bool:
        """
        Is message sent by current TG user
        Args:
            message: telegram message
        """
        return message["from_id"] == MY_ID

    def my_message_labels(self, messages: list[dict]) -> list[Message]:
        """
        Is message sent by current TG user
        Args:
            messages: telegram message
        """
        results = []
        for message in messages:
            role = self.get_role_from_message(message)
            results.append(Message(role=role, content=message["text"]))
        return results

    @staticmethod
    def filter_text(messages: list[dict]) -> list[dict]:
        """
        Filter all user message, remove links and etc
        Args:
            messages: telegram message
        """
        for message in messages:
            text = message["text"]
            text = TGBaseSocialReader.process_text(text)
            text = remove_links(text)
            text = re.sub(laugh_regex, r"\1ахах\4", text, flags=re.M)
            text = TGBaseSocialReader.merge_any_while_spaces(text)
            message["text"] = text
        messages = [message for message in messages if len(message["text"])]
        return messages

    def filter_users_messages(self, chats: list[dict]) -> list[list[dict]]:
        """
        Filter all user message, remove links and etc
        Args:
            chats: telegram message
        """
        chats = list(filter(self.filter_users_cond, chats))
        log.info(f"Filtered chat count %s", len(chats))

        filtered_messages = [
            self.filter_messages(chat["messages"])
            for chat in tqdm(chats, desc="Filter messages in every chat")
        ]
        log.info("Text preprocess")

        for user in tqdm(filtered_messages, desc="Preprocess text in every chat"):
            for msg in user:
                msg["text"] = self.process_text(msg["text"])

        filtered_messages = [
            TGBaseSocialReader.filter_text(chat)
            for chat in tqdm(filtered_messages, desc="Text preprocess")
        ]

        if self.debug:
            filtered_messages = filtered_messages[::52]

        filtered_messages = [self.remove_user_message_by_cond(user) for user in filtered_messages]
        filtered_messages = [sample for sample in filtered_messages if sample]

        return filtered_messages

    def prepare_data(self) -> list[list[Message]]:
        """Read data source, preprocess and return structured messages."""
        chats = self.read_chats()
        filtered_chats = self.filter_users_messages(chats)
        user_messages = [self.my_message_labels(message) for message in tqdm(filtered_chats)]
        return user_messages


class TGReplyOnlySocialReader(TGBaseSocialReader):
    def get_message_reply_pairs(self, messages: list[dict]) -> list[Message]:
        """
        Create QA dataset, where questions is messages of user and answer my reply to them
        Args:
            messages: user messages
        """
        message_dict = {}
        all_my_messages = []
        for message in messages:
            message_id = message["id"]
            message_dict[message_id] = message
            if self.is_my_message(message) and "reply_to_message_id" in message:
                all_my_messages.append(message)
        mapping = OrderedDict()
        for message in all_my_messages:
            reply_to_message_id = message["reply_to_message_id"]
            if reply_to_message_id not in message_dict:
                continue
            if reply_to_message_id in mapping:
                mapping[reply_to_message_id]["answers"].append(message)
            else:
                mapping[reply_to_message_id] = {
                    "question": message_dict[reply_to_message_id],
                    "answers": [message],
                }
        reply_dataset = []
        for key in mapping.keys():
            question = mapping[key]["question"]
            reply_dataset.append(
                Message(role=self.get_role_from_message(question), content=question["text"])
            )
            answer_text = "\n".join(item["text"] for item in mapping[key]["answers"])
            core_answer = deepcopy(mapping[key]["answers"][0])
            core_answer["text"] = answer_text
            reply_dataset.append(
                Message(role=self.get_role_from_message(core_answer), content=core_answer["text"])
            )

        return reply_dataset

    def prepare_data(self) -> list[list[Message]]:
        """Read data source, preprocess and return structured messages."""
        chats = self.read_chats()
        filtered_chats = self.filter_users_messages(chats)
        user_messages = [self.get_message_reply_pairs(message) for message in tqdm(filtered_chats)]
        return user_messages
