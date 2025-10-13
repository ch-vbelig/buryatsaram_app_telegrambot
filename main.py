import telebot
from telebot.apihelper import ApiException, ApiTelegramException
import config
import setup
import text_processing as txt_process
import synthesis
import message_replies as rpl
import logging

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

bot = telebot.TeleBot(setup.TOKEN)

CHAT_STATES = {}

# Possible states for a chat
WAITING_FOR_INPUT = 'waiting'
PROCESSING_REQUEST = 'processing'


def get_state(chat_id):
    return CHAT_STATES.get(chat_id, WAITING_FOR_INPUT)

def set_state(chat_id, state):
    CHAT_STATES[chat_id] = state


@bot.message_handler(commands=['start'])
def start_command(message):
    chat_id = message.chat.id
    set_state(chat_id, WAITING_FOR_INPUT)
    try:
        bot.send_message(message.chat.id, rpl.msg_start.format(message.from_user.first_name))
        bot.send_message(message.chat.id, rpl.msg_send_content)
    except ApiTelegramException as e:
        logging.critical(f"Caught Telegram Api Exception: {e}\nMessage from @{message.from_user.username}({message.from_user.first_name} {message.from_user.last_name}): {msg}")
    except ApiException as e:
        logging.critical(f"Caught Api Exception: {e}\nMessage from @{message.from_user.username}({message.from_user.first_name} {message.from_user.last_name}): {msg}")
    except Exception as e:
        logging.critical(f"Caught unknown exception: {e}\nMessage from @{message.from_user.username}({message.from_user.first_name} {message.from_user.last_name}): {msg}")
    else:
        logging.info(f"Message from @{message.from_user.username}({message.from_user.first_name} {message.from_user.last_name}): {msg}")



@bot.message_handler(content_types=['text'], func=lambda m: get_state(m.chat.id) == WAITING_FOR_INPUT)
def send_text(message):
    chat_id = message.chat.id
    msg = message.text.lower().strip()
    msg = txt_process.normalize(msg)

    try:
        if len(msg) <= 1:
            bot.send_message(message.chat.id, rpl.msg_too_short)
        elif len(msg) > config.max_msg_len:
            bot.send_message(message.chat.id, rpl.msg_too_long)
        elif txt_process.contains_curse_words(msg):
            bot.send_message(message.chat.id, rpl.msg_contains_abusive_words)
        else:
            set_state(chat_id, PROCESSING_REQUEST)
            start_synthesis(msg, message)
    except ApiTelegramException as e:
        logging.critical(f"Caught Telegram Api Exception: {e}\nMessage from @{message.from_user.username}({message.from_user.first_name} {message.from_user.last_name}): {msg}")
    except ApiException as e:
        logging.critical(f"Caught Api Exception: {e}\nMessage from @{message.from_user.username}({message.from_user.first_name} {message.from_user.last_name}): {msg}")
    except Exception as e:
        logging.critical(f"Caught unknown exception: {e}\nMessage from @{message.from_user.username}({message.from_user.first_name} {message.from_user.last_name}): {msg}")
    else:
        logging.info(f"Message from @{message.from_user.username}({message.from_user.first_name} {message.from_user.last_name}): {msg}")



def start_synthesis(msg, message):
    chat_id = message.chat.id
    bot.send_message(message.chat.id, rpl.msg_reading)
    voice = synthesis.synthesize(msg)
    bot.send_audio(message.chat.id, voice)
    # bot.delete_message(temp_msg.chat.id, temp_msg.id)
    set_state(chat_id, WAITING_FOR_INPUT)
    bot.send_message(message.chat.id, rpl.msg_send_content)


@bot.message_handler(content_types=['text'], func=lambda m: get_state(m.chat.id) == PROCESSING_REQUEST)
def block_request(message):
    bot.send_message(message.chat.id, rpl.msg_reading_in_process)


bot.polling()