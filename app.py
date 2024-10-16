from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

# Đọc dữ liệu từ file CSV
data = pd.read_csv('E:/PythonProjects/CommentsAnalysisApp/data/five_movies_comments_data.csv')






@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    selected_movie = ""  
    pie_chart = ""
    time_chart = ""  # Biến biểu đồ thời gian
    topics_display = ""  # Biến để lưu trữ các chủ đề LDA

    # Lấy danh sách các phim từ cột 'Movie' trong DataFrame
    movie_list = data['Movie'].unique()

    if request.method == 'POST':
        selected_movie = request.form.get('movie', selected_movie)
    else:
        selected_movie = movie_list[0]  # Chọn phim đầu tiên mặc định

    # Lọc dữ liệu dựa trên phim đã chọn
    results_df = data[data['Movie'] == selected_movie]

    # Kiểm tra xem có dữ liệu không
    if not results_df.empty:
        # Đếm số lượng cảm xúc cho biểu đồ tròn
        labels_count = results_df['Label'].value_counts().reset_index()
        labels_count.columns = ['sentiment', 'count']

        # Tạo biểu đồ tròn với Plotly
        fig = px.pie(labels_count, names='sentiment', values='count',
                     title=f'Phân bố cảm xúc cho phim {selected_movie}')
        pie_chart = pio.to_html(fig, full_html=False)

        # Chuyển đổi cột Published At sang định dạng datetime nếu chưa có
        results_df['Published At'] = pd.to_datetime(results_df['Published At'])

        # Nhóm dữ liệu theo tuần và cảm xúc
        time_series = results_df.groupby([pd.Grouper(key='Published At', freq='W'), 'Label']).size().reset_index(name='count')

        # Tạo biểu đồ thời gian cảm xúc với Plotly
        fig_time = px.line(time_series, x='Published At', y='count', color='Label',
                           title=f'Xu hướng cảm xúc theo thời gian cho phim {selected_movie}')
        time_chart = pio.to_html(fig_time, full_html=False)

        
        # Chuyển đổi DataFrame sang danh sách các từ điển
        results = results_df.to_dict('records')

    return render_template('index.html', results=results, selected_movie=selected_movie, 
                           pie_chart=pie_chart, time_chart=time_chart, 
                           movie_list=movie_list)  # Truyền danh sách phim vào template

if __name__ == '__main__':
    app.run(debug=True)
