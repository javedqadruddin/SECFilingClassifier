from keras.preprocessing.text import text_to_word_sequence
import os


# keras NLP tools filter out certain tokens by default
# this function replaces the default with a smaller set of things to filter out
def filter_not_punctuation():
    return '"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n'


def get_first_n_words(text, n):
    string_sequence = text_to_word_sequence(text, filters=filter_not_punctuation())
    truncated_string = ''
    for word in string_sequence[:n]:
        truncated_string = truncated_string + word + ' '
    return truncated_string




# gets text data from files with only maxlen words from each file. Gets whole file if maxlen is None
def get_labelled_data_from_directories(data_dir, maxlen=None):
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                f = open(fpath)
                t = f.read()
                if maxlen is not None:
                    t = get_first_n_words(t, maxlen)
                texts.append(t)
                f.close()
                labels.append(label_id)
    return texts, labels_index, labels
