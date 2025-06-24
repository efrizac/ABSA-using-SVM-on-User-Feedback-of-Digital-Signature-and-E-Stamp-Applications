from flask import Flask, request, jsonify, render_template
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import pandas as pd
import re
import pickle
from io import StringIO
from flask import Flask, request, jsonify, render_template
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO

clf = pd.read_pickle('model_svc_mlros_70.pkl')
vectorizer = pd.read_pickle('tfidf_vectorizer.pkl')
normalizad_word = pd.read_csv('kamusalay.csv', usecols=['slang', 'formal'])

# Inisialisasi Flask
app = Flask(__name__)

# Membuat dictionary normalisasi
normalizad_word_dict = dict(zip(normalizad_word['slang'], normalizad_word['formal']))


# Load stopwords
stopwords_list = set(stopwords.words('indonesian'))
stopwords_list.update(["aplikasi", "privy", "privyid", "vida", "xignature", "akulaku"])

# Initialize stemmer from Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for text preprocessing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords_list, normalizad_word_dict, stemmer):
        self.stopwords_list = stopwords_list
        self.normalizad_word_dict = normalizad_word_dict
        self.stemmer = stemmer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text):
        # Text cleaning
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenization
        tokens = word_tokenize(text)

        # Normalization
        tokens = [self.normalizad_word_dict.get(term, term) for term in tokens]

        # Stopword removal
        tokens = [token for token in tokens if token not in self.stopwords_list]

        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)

# Use the custom transformer in the pipeline
text_preprocessor = TextPreprocessor(stopwords_list, normalizad_word_dict, stemmer)

text_pipeline = Pipeline([
    ('preprocess', text_preprocessor)
])

model_pipeline = Pipeline([
    ('preprocessing', text_pipeline),
    ('vectorizer', vectorizer),
    ('classifier', clf)
])

# Pipeline model yang digunakan untuk predict
model = model_pipeline

# Rename label
def rename_labels(value):
    if value == 0:
        return "Tidak Relevan"
    elif value == 1:
        return "Negatif"
    elif value == 2:
        return "Positif"
    return value 

# Mapping aspek dan label sentimen
aspek_mapping = ["Login dan Verifikasi", "Efisiensi", "Layanan Pengguna", "Responsivitas"]
sentimen_mapping = {1: "Negatif", 2: "Positif", 0: "Tidak relevan"}

def predict_mapping(predictions):
    hasil = []
    for i, pred in enumerate(predictions):
        if pred != 0:  # Hanya aspek yang relevan
            hasil.append({
                "aspek": aspek_mapping[i],
                "sentimen": sentimen_mapping[pred]
            })
    return hasil

# def create_sentiment_per_aspect_per_year(data):
#     aspect_columns = ['Login dan Verifikasi', 'Efisiensi', 'Layanan Pengguna', 'Responsivitas']
    
#     # Pastikan kolom 'at' ada dan ekstrak tahun
#     if 'at' in data.columns:
#         try:
#             data['year'] = pd.to_datetime(data['at']).dt.year
#         except:
#             # Jika konversi gagal, gunakan tahun saat ini
#             data['year'] = pd.Timestamp.now().year
#     else:
#         # Jika tidak ada kolom timestamp, gunakan tahun saat ini
#         data['year'] = pd.Timestamp.now().year
    
#     # Filter hanya data yang relevan (positif atau negatif)
#     relevant_data = data[data[aspect_columns].isin(['Positif', 'Negatif']).any(axis=1)]
    
#     # Buat subplots
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
#     axes = axes.flatten()
#     plt.tight_layout(pad=5.0)
    
#     for i, aspect in enumerate(aspect_columns):
#         ax = axes[i]
        
#         # Group by year dan sentimen untuk aspek saat ini
#         aspect_data = relevant_data[relevant_data[aspect].isin(['Positif', 'Negatif'])]
        
#         # Jika tidak ada data untuk aspek ini, buat plot kosong
#         if aspect_data.empty:
#             ax.text(0.5, 0.5, 'Tidak ada data', 
#                    ha='center', va='center', fontsize=12)
#             ax.set_title(f'Sentimen {aspect} per Tahun', fontsize=12)
#             continue
            
#         grouped = aspect_data.groupby(['year', aspect]).size().unstack(fill_value=0)
        
#         # Plot stacked bar chart
#         grouped.plot(kind='bar', stacked=True, ax=ax, color=['#ff7f0e', '#1f77b4'])
        
#         ax.set_title(f'Sentimen {aspect} per Tahun', fontsize=12)
#         ax.set_xlabel('Tahun', fontsize=10)
#         ax.set_ylabel('Jumlah Ulasan', fontsize=10)
#         ax.legend(title='Sentimen', bbox_to_anchor=(1.05, 1), loc='upper left')
        
#         # Tambahkan anotasi
#         for p in ax.containers:
#             ax.bar_label(p, label_type='center', fmt='%d', color='white', fontsize=8)
    
#     # Konversi plot ke gambar base64
#     img = BytesIO()
#     plt.savefig(img, format='png', bbox_inches='tight')
#     img.seek(0)
#     chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
#     plt.close(fig)
    
#     return f'<img src="data:image/png;base64,{chart_base64}" alt="Sentimen per Aspek per Tahun" width="900"/>'

import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def create_sentiment_per_aspect_per_year(data):
    aspect_columns = ['Login dan Verifikasi', 'Efisiensi', 'Layanan Pengguna', 'Responsivitas']
    
    # Pastikan kolom 'at' ada dan ekstrak tahun
    if 'at' in data.columns:
        try:
            data['year'] = pd.to_datetime(data['at']).dt.year
        except:
            data['year'] = pd.Timestamp.now().year
    else:
        data['year'] = pd.Timestamp.now().year
    
    # Filter hanya data yang relevan (positif atau negatif)
    relevant_data = data[data[aspect_columns].isin(['Positif', 'Negatif']).any(axis=1)]
    
    # Buat subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()
    
    for i, aspect in enumerate(aspect_columns):
        ax = axes[i]
        
        # Group by year dan sentimen untuk aspek saat ini
        aspect_data = relevant_data[relevant_data[aspect].isin(['Positif', 'Negatif'])]
        
        if aspect_data.empty:
            ax.text(0.5, 0.5, 'Tidak ada data', 
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'Sentimen {aspect} per Tahun', fontsize=12)
            continue
            
        grouped = aspect_data.groupby(['year', aspect]).size().unstack(fill_value=0)
        
        grouped.plot(kind='bar', stacked=True, ax=ax, color=['#ff7f0e', '#1f77b4'])
        
        ax.set_title(f'Sentimen {aspect} per Tahun', fontsize=12)
        ax.set_xlabel('Tahun', fontsize=10)
        ax.set_ylabel('Jumlah Ulasan', fontsize=10)
        
        for p in ax.containers:
            ax.bar_label(p, label_type='center', fmt='%d', color='white', fontsize=8)

    # Tambahkan legend global
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Sentimen', loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    
    # Konversi plot ke gambar base64
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return f'<img src="data:image/png;base64,{chart_base64}" alt="Sentimen per Aspek per Tahun" width="900"/>'


def create_wordcloud_for_aspects(data, aspect_columns):
    # Menyimpan hasil WordCloud untuk setiap aspek
    wordcloud_images_html = ""
    
    for column in aspect_columns:
        # Gabungkan semua teks dalam kolom saat nilai aspek relevan (misalnya: label positif atau negatif)
        relevant_text = data[data[column] != 0]['Ulasan'].dropna()
        text = " ".join(review for review in relevant_text)

        # Buat WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

        # Konversi gambar ke Base64
        img = BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

        # Tambahkan WordCloud ke dalam HTML dengan judul kolom
        wordcloud_images_html += f'<h3>WordCloud for {column}</h3>'
        wordcloud_images_html += f'<img src="data:image/png;base64,{img_base64}" alt="WordCloud for {column}" width="800" height="400"/>'
        wordcloud_images_html += '<br><br>'
    
    return wordcloud_images_html

def create_distribution_chart(data):
    aspect_columns = ['Login dan Verifikasi', 'Efisiensi', 'Layanan Pengguna', 'Responsivitas']
    aspect_labels = ['Login dan Verifikasi', 'Efisiensi', 'Layanan Pengguna', 'Responsivitas']
    aspect_counts = []

    for col in aspect_columns:
        # Filter data untuk hanya menyertakan label "Positif" dan "Negatif"
        filtered_data = data[data[col].isin(["Positif", "Negatif"])]
        counts = filtered_data[col].value_counts()
        aspect_counts.append(counts)

    # Buat subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    axes = axes.flatten()
    plt.tight_layout(pad=6.0)  # Tambahkan padding antar subplot

    # Plot setiap aspek
    for i, (counts, label) in enumerate(zip(aspect_counts, aspect_labels)):
        ax = axes[i]
        counts.plot(kind='bar', ax=ax, color=["#1f77b4", "#ff7f0e"])
        ax.set_title(label)
        ax.set_xlabel("Sentimen")
        ax.set_ylabel("Jumlah")
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Konversi plot ke gambar base64
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)  # Tutup plot setelah disimpan

    # Kembalikan HTML gambar
    return f'<img src="data:image/png;base64,{chart_base64}" alt="Distribution Chart" width="800"/>'


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    text = data.get('text', '')  # Ambil input teks dari form
    predictions = model.predict([text])[0]

    # Olah prediksi ke format informasi
    hasil = predict_mapping(predictions)
    return render_template('text-output.html', text=text, hasil=hasil)

@app.route('/upload', methods=['POST'])
def upload_file_route():
    uploaded_file = request.files.get('file')

    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    # Validasi format file
    if not uploaded_file.filename.endswith(('.txt', '.csv')):
        return jsonify({"error": "Invalid file format. Only .txt and .csv are allowed."}), 400

    # Baca file
    file_content = uploaded_file.read().decode('utf-8')

    # Proses file CSV
    if uploaded_file.filename.endswith('.csv'):
        data = pd.read_csv(StringIO(file_content))
    else:
        data = pd.DataFrame(file_content.splitlines(), columns=['review'])

    # Validasi kolom 'review'
    if 'review' not in data.columns:
        return jsonify({"error": "Missing 'review' column in the file."}), 400

    # Prediksi dengan model
    predictions = model.predict(data['review'])

    # Tambahkan kolom aspek
    for i, aspect in enumerate(['Login dan Verifikasi', 'Efisiensi', 'Layanan Pengguna', 'Responsivitas']):
        data[aspect] = [sentimen_mapping[pred[i]] for pred in predictions]

    # Rename kolom jika diperlukan
    data.rename(columns={'review': 'Ulasan'}, inplace=True)

    # Buat WordCloud dan Distribusi
    aspect_columns = ['Login dan Verifikasi', 'Efisiensi', 'Layanan Pengguna', 'Responsivitas']

    return render_template(
        'table.html',
        table=data.to_html(classes='table table-striped', index=False),
        wordcloud_html=create_wordcloud_for_aspects(data, aspect_columns),
        year_based_distribution_chart_html=create_sentiment_per_aspect_per_year(data),
        distribution_chart_html=create_distribution_chart(data)
        # table=data.to_html(classes='table table-striped', index=False)
    )

# @app.route('/display_results')
# def display_results():
#     # Data dummy, gantikan dengan dataset sebenarnya
#     data = ...
#     aspect_columns = ['Login dan Verifikasi', 'Efisiensi', 'Layanan Pengguna', 'Responsivitas']

#     # Hasilkan WordCloud dan Distribution Chart
#     wordcloud_html = create_wordcloud_for_aspects(data, aspect_columns)
#     distribution_chart_html = create_distribution_chart(data)

#     # Konversikan tabel ke HTML
#     table_html = data.to_html(classes='table table-striped', index=False)

#     # Render template dengan WordCloud, Distribution Chart, dan tabel
#     return render_template('table.html', wordcloud_html=wordcloud_html, distribution_chart_html=distribution_chart_html, table_html=table_html)

if __name__ == '__main__':
    app.run(debug=True)


# %%
print(distribution_chart_html)  # Harus berupa string HTML <img>
print(wordcloud_html)          # Harus berupa string HTML <img>
