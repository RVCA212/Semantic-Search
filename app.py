from flask import Flask, render_template, request, jsonify
import semantic_search

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        top_k = int(request.form.get('top_k', 5))
        results = semantic_search.search(query, top_k)
        return jsonify(results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
