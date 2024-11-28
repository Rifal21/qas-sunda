from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
from gensim.models import Word2Vec

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

# Logo path
logo_path = os.path.join(app.config['STATIC_FOLDER'], 'pananyaan abah.jpg')

# Langkah 1: Persiapan Data
data = pd.read_csv('QASSND50k.csv')

# Langkah 2: Pra-Pemrosesan Data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['pertanyaan'])
X = tokenizer.texts_to_sequences(data['pertanyaan'])
X = pad_sequences(X)
context = data['konteks']
y = data['jawaban'].apply(lambda x: 1 if x == 'leres' else 0)
X_train, X_test, context_train, context_test, y_train, y_test = train_test_split(X, context, y, test_size=0.2, random_state=42)

# Membuat model Word2Vec untuk konteks
context_tokens = [tokenizer.texts_to_sequences([sentence])[0] for sentence in context]
word2vec_model = Word2Vec(sentences=context_tokens, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(context_tokens, total_examples=len(context_tokens), epochs=10)

# Membuat vektor embedding konteks
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# Langkah 3: Membangun Model LSTM untuk Menilai Kesesuaian Pertanyaan dan Konteks
model_file = 'model_lstm.h5'

if os.path.exists(model_file):
    # Memuat model jika sudah ada
    model = load_model(model_file)
else:
    # Jika model belum ada, lakukan pelatihan dan simpan model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

    # Simpan model setelah pelatihan
    model.save(model_file)

# List untuk menyimpan pertanyaan, jawaban, dan konteks yang sudah terjawab
answered_questions = []

# Langkah 4: Memprediksi Jawaban dan Konteks
# (implementasikan seperti sebelumnya)
def predict_answer(question, threshold=0.5):
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq = pad_sequences(question_seq, maxlen=X.shape[1])

    # Menggunakan model untuk memprediksi apakah jawaban adalah "ya" atau "tidak"
    predicted_result = model.predict(question_seq)
    if predicted_result >= threshold:
        answer = "leres"
    else:
        answer = "henteu"

    # Menganalisis pertanyaan untuk mencocokannya dengan konteks yang sesuai
    best_context = ""
    max_similarity = -1.0

    for i, context_text in enumerate(context_train):
        similarity = compute_similarity(question, context_text)
        if similarity > max_similarity:
            max_similarity = similarity
            best_context = context_text

    # Pilih jawaban dari dataset berdasarkan konteks yang sesuai
    if max_similarity >= threshold:
        answer = data[data['konteks'] == best_context]['jawaban'].values[0]

    # Menambahkan pertanyaan, jawaban, dan konteks ke daftar yang sudah terjawab
    answered_questions.append({'question': question, 'answer': answer, 'context': best_context})

    return answer, best_context


# def compute_similarity(question, context):
#     question_words = set(question.lower().split())
#     context_words = set(context.lower().split())

#     # Hitung jumlah kata yang cocok antara pertanyaan dan konteks
#     matched_words = question_words.intersection(context_words)

#     return len(matched_words)

def compute_similarity(question, context):
    question_words = set(question.lower().split())
    context_words = set(context.lower().split())

    intersection = question_words.intersection(context_words)
    union = question_words.union(context_words)

    if len(union) == 0:
        return 0.0
    else:
        return len(intersection) / len(union)


@app.route('/')
def home():
    return render_template('index.html', answered_questions=answered_questions , gambar_logo=logo_path)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        question = request.form['question']
        answer, context = predict_answer(question)
        return render_template('index.html', question=question, answer=answer, context=context, answered_questions=answered_questions)


if __name__ == '__main__':
    app.run(debug=True)
