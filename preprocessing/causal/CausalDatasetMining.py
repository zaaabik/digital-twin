import re
from functools import partial
from multiprocessing import Pool

import datasets
from datasets import DatasetDict
from preprocessing.utils.social_reader.BaseSocialReader import BaseSocialReader, Message
from preprocessing.utils.social_reader.TG import BOT_ROLE, USER_ROLE
from preprocessing.utils.utils import get_dataset_statistic
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset.Chat import Conversation
from src.dataset.Dataset import ChatDataset

greetings_regexp = (
    "((п|П)ривет)|((д|Д)обрый день)|((к|К)укусики)|((к|К)уку)|((д|Д)оброе утро)|((д|Д)обрый день)"
    "|((д|Д)аров) "
)


class CausalDataset:
    """Base class to prepare list of user message."""

    filtered_chats: list[list[Message]] = None
    merged_message_chats: list[list[Message]] = None
    causal_dataset: list[list[Message]]

    def __init__(
        self,
        data_reader: BaseSocialReader,
        tokenizer_name: str,
        conversation: Conversation,
        num_workers: int = 8,
        debug: bool = False,
        max_token_count: int = 512,
        split_by_greeting: bool = True,
    ):
        super().__init__()
        self.data_reader = data_reader
        self.debug = debug
        self.tokenizer_name = tokenizer_name
        self.conversation: Conversation = conversation
        self.max_token_count = max_token_count
        self.split_by_greeting = split_by_greeting

        if debug:
            self.num_workers = 1
        else:
            self.num_workers = num_workers

    def merge_messages_for_all_chats(self, filtered_chats: list[list[Message]]):
        """
        If user has more than 1 message in a row, we merge it into singe message split by new line char
        Args:
            filtered_chats: users
        """
        causal_dataset = [self.merge_messages(chat) for chat in filtered_chats]
        return causal_dataset

    @staticmethod
    def remove_empty_msg(message: Message) -> int:
        """
        Check length of message
        Args:
            message: users
        """
        return len(message.content.strip())

    @staticmethod
    def merge_messages(user_messages: list[Message]):
        """
        If user has more than 1 message in a row, we merge it into singe message split by new line char
        Args:
            user_messages: user messages
        """
        user_messages = list(filter(CausalDataset.remove_empty_msg, user_messages))
        if len(user_messages) < 2:
            return []
        first_sample = user_messages[0]
        current_prefix = first_sample.role
        current_text = first_sample.content
        rows = []
        last_prefix = None
        for idx in range(1, len(user_messages)):
            current_sample = user_messages[idx]
            prefix = current_sample.role
            text = str(current_sample.content)
            text = text.strip()
            last_prefix = prefix

            if prefix != current_prefix:
                rows.append(Message(role=current_prefix, content=current_text))

                current_text = current_sample.content
                current_prefix = prefix
            else:
                current_text = str(current_text) + f"\n{text}"

        if last_prefix == current_prefix:
            rows.append(Message(role=current_prefix, content=current_text))
        if rows[-1].role == USER_ROLE:
            del rows[-1]
        elif re.findall(greetings_regexp, rows[-1].content):
            del rows[-1]
            if len(rows):
                del rows[-1]
        return rows

    @staticmethod
    def check_sides_count(messages: list[Message]):
        """
        Return if in conversation present only one side
        Args:
            messages: user messages
        """
        sides = set()
        for message in messages:
            sides.add(message.role)
        return len(sides) == 2

    def format_output(self, messages: list[Message], tokenizer) -> list[list[Message]]:
        """
        If length of conversation longer than max token length based on tokenizer we split it into parts
        Args:
            messages: user messages
            tokenizer: model tokenizer
        """
        total_output = []
        output = []
        for idx, message in enumerate(messages):
            current_msg = Message(role=message.role, content=message.content)
            if len(output) == 0 and current_msg.role == BOT_ROLE:
                continue

            conversation = self.conversation()
            conversation.expand(output + [current_msg])
            prompt = conversation.get_prompt(tokenizer)
            current_token_count = len(tokenizer(prompt)["input_ids"])

            if current_token_count > self.max_token_count:
                if self.check_sides_count(output):
                    total_output.append(output)

                conversation = self.conversation()
                conversation.expand([current_msg])
                prompt = conversation.get_prompt(tokenizer)
                current_token_count = len(tokenizer(prompt)["input_ids"])
                if current_token_count < self.max_token_count and current_msg.role != BOT_ROLE:
                    output = [current_msg]
                else:
                    output = []
            else:
                output = output + [current_msg]

        if len(output):
            if self.check_sides_count(output):
                total_output.append(output)

        return total_output

    def split_messages_by_max_seq_len(self, messages: list[list[Message]]) -> list[list[Message]]:
        """
        Run split function by length for users
        Args:
            messages: users
        """
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        format_output_with_tokenizer = partial(self.format_output, tokenizer=tokenizer)
        with Pool(self.num_workers) as pool:
            template_dialog = list(
                tqdm(
                    pool.imap(format_output_with_tokenizer, messages),
                    total=len(messages),
                    desc="Split messages by max token count",
                )
            )
        template_dialog_results = []
        for dialog_messages in template_dialog:
            template_dialog_results.extend(dialog_messages)

        template_dialog = [msg for msg in template_dialog_results if len(msg)]
        return template_dialog

    @staticmethod
    def split_chat_by_greetings(messages: list[Message]) -> list[list[Message]]:
        """
        We assume if we write greetings, in this case we need to reset context
        Args:
            messages: user messages
        """
        new_chats = []
        current_chat = []
        for message in messages:
            if message.role == USER_ROLE and re.findall(greetings_regexp, message.content):
                if current_chat:
                    new_chats.append(current_chat)
                current_chat = [message]
            else:
                current_chat.append(message)
        if len(current_chat) > 1:
            new_chats.append(current_chat)
        return new_chats

    def run(self) -> list[dict]:
        """Start preprocessing data."""
        filtered_chats = self.data_reader.prepare_data()

        print("Before split: ", len(filtered_chats))

        if self.split_chat_by_greetings:
            split_chats = []
            for user in filtered_chats:
                user_split_msgs = self.split_chat_by_greetings(user)
                split_chats.extend(user_split_msgs)
        else:
            split_chats = filtered_chats

        print("After split: ", len(split_chats))

        merged_messages = self.merge_messages_for_all_chats(split_chats)
        template_dialog = self.split_messages_by_max_seq_len(messages=merged_messages)

        template_dialog_json = [
            [{"role": message.role, "content": message.content} for message in user]
            for user in template_dialog
        ]
        return [{"messages": msg} for msg in template_dialog_json]


def build_hf_dataset_from_list(
    template_dialog: list[Message],
    tokenizer_name: str,
    conversation: Conversation,
    max_token_count: int,
) -> DatasetDict:
    """
    Create dataset from list of dialogs
    Args:
        max_token_count: max length of sequence
        conversation: conversation class
        tokenizer_name:
        template_dialog:
    """
    ds = datasets.Dataset.from_list(template_dialog)
    ds = ds.train_test_split(test_size=0.15)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_dataset = ChatDataset(
        original_records=list(ds["train"]),
        conversation=conversation,
        tokenizer=tokenizer,
        max_tokens_count=max_token_count,
        debug=False,
    )

    val_dataset = ChatDataset(
        original_records=list(ds["test"]),
        conversation=conversation,
        tokenizer=tokenizer,
        max_tokens_count=max_token_count,
        debug=False,
    )

    print("## Train dataset ###")
    get_dataset_statistic(train_dataset)

    print("## Val dataset ###")
    get_dataset_statistic(val_dataset)
    return ds
