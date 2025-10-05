import telebot
import config
import setup
import text_processing as txt_process
import synthesis
import message_replies as rpl


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
    bot.send_message(message.chat.id, rpl.msg_start.format(message.from_user.first_name))
    bot.send_message(message.chat.id, rpl.msg_send_content)


@bot.message_handler(content_types=['text'], func=lambda m: get_state(m.chat.id) == WAITING_FOR_INPUT)
def send_text(message):
    chat_id = message.chat.id
    msg = message.text.lower().strip()
    msg = txt_process.normalize(msg)

    if len(msg) <= 1:
        bot.send_message(message.chat.id, rpl.msg_too_short)
    elif len(msg) > config.max_msg_len:
        bot.send_message(message.chat.id, rpl.msg_too_long)
    elif txt_process.contains_curse_words(msg):
        bot.send_message(message.chat.id, rpl.msg_contains_abusive_words)
    else:
        set_state(chat_id, PROCESSING_REQUEST)
        start_synthesis(msg, message)


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