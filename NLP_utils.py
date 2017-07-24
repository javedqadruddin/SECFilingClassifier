from keras.preprocessing.text import text_to_word_sequence
import os

# generates batches of training data infinitely of given size (though last batch before sequence repeats may be smaller)
def gen_batches(batch_size, x_vals, y_vals):
    start = 0
    while True:
        yield (x_vals[start:start+batch_size], y_vals[start:start+batch_size])
        if(start + batch_size < len(x_vals)):
            start += batch_size
        else:
            start = 0


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
