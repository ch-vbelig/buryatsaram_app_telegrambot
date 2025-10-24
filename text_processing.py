import re
from config import Config

def open_file(path):
    with open(path, encoding='utf-8') as fp:
        words = [w.strip() for w in fp.readlines()]
    return words

CURSE_WORDS_PATH = 'ru_curse_words_stems.txt'
curse_words = open_file(CURSE_WORDS_PATH)
curse_words_pattern = '|'.join(curse_words)

def contains_curse_words(text):
    words = text.split()
    for word in words:
        for curse in curse_words:
            if word.startswith(curse):
                return True
    return False

def vocab_lookup(text):
    vocab_char_to_idx = {char: idx for idx, char in enumerate(Config.vocab)}
    return [vocab_char_to_idx[char] for char in text]

def vocab_decipher(ids):
    vocab_idx_to_char = {idx: char for idx, char in enumerate(Config.vocab)}
    return ''.join([vocab_idx_to_char[idx] for idx in ids])

def normalize(text):
    text = text.lower()
    text = re.sub(r'^\W*', '', text)
    text = text.strip()

    text = re.sub('\n\n', '.', text)
    text = re.sub('\n', ',', text)

    # substitute characters 'h' and 'y' with 'һ' and 'ү'
    text = re.sub('h', 'һ', text)
    text = re.sub('y', 'ү', text)
    text = re.sub('цы', 'сэ', text)
    text = re.sub('ц', 'с', text)
    text = re.sub(r' (даа|бэ|лэ|лээ|гу|гү|бшу|бшуу|абза)(б|бди|ш|т)*(\W+)', r'\1\2\3', text)
    text = re.sub(r' (\w)$', r'\1.', text)

    # remove non-alphabetic characters
    pattern = f'[^{Config.vocab_bur_only}]'
    text = re.sub(pattern, '', text)

    # remove double spaces
    text = re.sub(r'\s+\s+', ' ', text)
    text = re.sub(r'\s([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])\s*[.,!?]*', r'\1', text)

    text = text.strip()
    return text

def normalize_long_text(text, chunk_size=4, verbose=False):
    normalized_text = normalize(text)
    pattern = f'([{Config.sounds_only}]*[.,!?])'
    sintagmas = [s.strip() for s in re.split(pattern, normalized_text) if len(s)>1]

    sequences = []

    for s in sintagmas:
        words = s.split()
        chunks = [' '.join(words[i:i+chunk_size]).strip() + Config.vocab_end_of_text for i in range(0, len(words), chunk_size)]
        sequences.extend(chunks)

    if verbose:
        print(sequences)

    id_batch = []
    for seq in sequences:
        ids = vocab_lookup(seq)
        id_batch.append(ids)

    return id_batch, sequences


if __name__ == '__main__':
    t = 'ывфа,, фа'
    t = normalize_long_text(t, verbose=True)
    print(t)