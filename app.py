from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 事前に作成したbeans.csvをロードしておく
df_beans = pd.read_csv("beans.csv")
feature_cols = ['bitterness','acidity','body','sweetness','aroma','aftertaste']

@app.route('/')
def index():
    # DataFrame の先頭 3 行を取り出し、
    # 各行を辞書 { 'name': ..., 'description': ... } のリストに変換
    beans_list = df_beans.head(3).to_dict('records')
    return render_template('index.html', beans=beans_list)

@app.route('/diagnosis/', methods=['GET','POST'])
def diagnosis():
    if request.method == 'POST':
        # スライダー値を受け取る
        vals = [ float(request.form[f's{idx}']) for idx in range(1,7) ]
        user_vec = np.array([vals])
        X = df_beans[feature_cols].to_numpy()
        sims = cosine_similarity(user_vec, X)[0]
        # MinMax正規化する場合はここで加工してもOK
        df_beans['score'] = (sims - sims.min())/(sims.max()-sims.min())*100
        top = df_beans.nlargest(1, 'score').iloc[0]
        return render_template('result.html', bean=top)
    # GET時はスライダー入力画面を表示
    return render_template('diagnosis.html')

if __name__ == '__main__':
    app.run(debug=True)
