import os
import pickle
import json
import re
import unicodedata
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. REPLICATE YOUR CLASSES FROM COLAB ---
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, 
                                        return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_size=1):
        return tf.zeros((batch_size, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, 
                                        return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        return self.fc(output), state, attention_weights

# --- 2. CONFIGURATION & LOADING ---
SAVE_DIR = "nmt_en_de_model"
app = Flask(__name__)
CORS(app) # Vital for connecting with your HTML file

# Load assets
with open(os.path.join(SAVE_DIR, 'inp_tokenizer.pkl'), 'rb') as f:
    inp_lang = pickle.load(f)
with open(os.path.join(SAVE_DIR, 'targ_tokenizer.pkl'), 'rb') as f:
    targ_lang = pickle.load(f)
with open(os.path.join(SAVE_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

# Instantiate and Load Weights
encoder = Encoder(config['vocab_inp_size'], config['embedding_dim'], config['units'], 1)
decoder = Decoder(config['vocab_tar_size'], config['embedding_dim'], config['units'], 1)

# Dummy call to initialize variables before loading weights
dummy_input = tf.zeros((1, config['max_length_inp']))
hidden = encoder.initialize_hidden_state(1)
enc_out, enc_hidden = encoder(dummy_input, hidden)
decoder(tf.expand_dims([targ_lang.word_index['<start>']], 0), enc_hidden, enc_out)

encoder.load_weights(os.path.join(SAVE_DIR, 'encoder_weights.weights.h5'))
decoder.load_weights(os.path.join(SAVE_DIR, 'decoder_weights.weights.h5'))

# --- 3. INFERENCE LOGIC ---
def preprocess_sentence(w):
    w = ''.join(c for c in unicodedata.normalize('NFD', w.lower().strip()) if unicodedata.category(c) != 'Mn')
    w = re.sub(r"([?.!,¡])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Zäöüß?.!,]+", " ", w).strip()
    return f'<start> {w} <end>'

@app.route('/translate', methods=['POST'])
def translate():
    sentence = request.json.get('text', '')
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ') if i in inp_lang.word_index]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=config['max_length_inp'], padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, config['units']))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for _ in range(config['max_length_targ']):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = targ_lang.index_word[predicted_id]
        if word == '<end>': break
        result += word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)

    return jsonify({"german": result.strip()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)