import os
import re
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
from bs4 import BeautifulSoup, Tag
from preprocessing.utils.social_reader.BaseSocialReader import BaseSocialReader, Message
from preprocessing.utils.social_reader.TG import laugh_regex
from preprocessing.utils.utils import remove_links
from tqdm import tqdm

from src.utils.logger import get_pylogger

RESTRICTED_RULES_REGEX = r"-"  # remove group chats
FILTER_GROUP_CHATS_MAX_ID = 2000000000

BOT_ROLE = "bot"
USER_ROLE = "user"

ref_regex = "(\n?(Ссылка|Видеозапись|Запись на стене|Аудиозапись|Файл|Стикер|Фотография)\n?)"
smile_regex = ":3+"
merge_many_nl_and_spaces = "[\n| ]{2,}"
attached_msg_regex = "(\n?\\d+ прикрепл(ё|е)нн(ых|ое) (сообщений|сообщения|сообщение)\n?)"
only_special_chars = "[а-яёА-ЯЁa-zA-Z0-9]"

log = get_pylogger(__name__)


def replacer(message, age_answer_regexp: str, new_age_answer: str):
    return re.sub(age_answer_regexp, new_age_answer, message)


def content_process(content):
    if isinstance(content, Tag) and content.name == "br":
        return "\n"
    else:
        return str(content)


def filter_msg(message):
    role = message.find("div", {"class": "message__header"})
    element = message.find("div", {"class": None})
    kludges = element.find("div", {"class": "kludges"})
    if kludges:
        kludges.decompose()
    message = "".join(content_process(x) for x in element.contents)

    name = role.text.split(",")[0]
    time = role.text[role.text.find(",") + 1 :].lstrip(" ").rstrip(" ")
    date = time[: time.find(" в ")]
    date = int(date.split(" ")[-1])

    return {"text": message, "date": date, "sender": name}


def get_number(file):
    number = re.findall(
        r"\d+",
        os.path.basename(file),
    )[0]
    return int(number)


def sort_user_files(files):
    indexes = np.argsort([get_number(file) for file in files])
    indexes = indexes[::-1]
    return [files[index] for index in indexes]


def get_messages(path):
    with open(path, encoding="windows-1251") as fp:
        soup = BeautifulSoup(fp, "html.parser")

    messages = soup.find("div", {"class": "wrap_page_content"})
    messages = messages.find_all("div", {"class": "message"})

    messages = [filter_msg(message) for message in messages]

    messages = messages[::-1]  # reverse to make chronological order

    return messages


def get_all_messages_by_user_path(user_path: str):
    files = glob(str(Path(user_path, "*")))
    files = sort_user_files(files)  # sort files with messages by a time
    try:
        messages = []
        for file in files:
            messages.extend(get_messages(file))
        return messages
    except Exception as e:
        print(e)
        return []


def filter_function_chat_folder(chat_folder):
    if len(re.findall(RESTRICTED_RULES_REGEX, chat_folder)):
        return False
    folder_name = os.path.basename(chat_folder)
    vk_id = int(folder_name)
    return vk_id < FILTER_GROUP_CHATS_MAX_ID


def get_user(base_folder: str):
    all_chat_folders = glob(os.path.join(base_folder, "*"))
    return all_chat_folders


def remove_messages_by_year_thr(user_messages: List, year: int):
    return list(filter(lambda x: x["date"] > year, user_messages))


def merge_any_while_spaces(text: str):
    any_whitespace_regexp = "[\t ]+"
    text = re.sub(any_whitespace_regexp, " ", text)
    any_new_line_regexp = "\n+"
    text = re.sub(any_new_line_regexp, "\n", text)
    return text


def filter_text(user_messages):
    for user_message in user_messages:
        message = remove_links(user_message["text"])
        message = re.sub(attached_msg_regex, " ", message)
        message = re.sub(ref_regex, " ", message)
        message = re.sub(laugh_regex, r"\1ахах\4", message, flags=re.M)
        message = re.sub(smile_regex, "", message)
        user_message["text"] = message

    user_messages = [user_message for user_message in user_messages if len(user_message["text"])]
    return user_messages


class VKBaseSocialReader(BaseSocialReader):
    def is_my_message(self, message: dict) -> bool:
        return message["sender"] == "Вы"

    chats: list[list[dict]]

    def __init__(self, min_years: int, **kwargs):
        super().__init__(**kwargs)
        self.data_path = os.path.join(self.data_path, "messages")
        self.min_years = min_years

    def read_chats(self):
        all_chat_folder = get_user(self.data_path)
        filtered_chat_folders = list(filter(filter_function_chat_folder, all_chat_folder))

        if self.debug:
            filtered_chat_folders = filtered_chat_folders[::52]

        with Pool(self.num_workers) as p:
            total_messages = tqdm(
                p.imap(get_all_messages_by_user_path, filtered_chat_folders),
                total=len(filtered_chat_folders),
            )
            return list(total_messages)

    def filter_chats(self, chats):
        total_messages_filtered_by_year = [
            remove_messages_by_year_thr(chat, self.min_years) for chat in chats
        ]
        total_messages_filtered_by_year = [
            user for user in total_messages_filtered_by_year if len(user)
        ]
        log.info(
            f"Filtered chat count remove old messages %s", len(total_messages_filtered_by_year)
        )

        total_messages_filtered_by_year_filtered_text = [
            filter_text(user) for user in total_messages_filtered_by_year
        ]
        filtered_chats = [
            user for user in total_messages_filtered_by_year_filtered_text if len(user)
        ]

        filtered_chats = [self.remove_user_message_by_cond(user) for user in filtered_chats]
        return filtered_chats

    def prepare_data(self) -> list[list[Message]]:
        """Read data source, preprocess and return structured messages."""
        chats = self.read_chats()
        filtered_chats = self.filter_chats(chats)
        user_messages = [self.my_message_labels(message) for message in tqdm(filtered_chats)]

        return user_messages
