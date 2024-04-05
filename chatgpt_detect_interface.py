import gradio as gr
import torch
from torch.nn.functional import softmax
import shap
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer,RobertaForSequenceClassification, pipeline,RobertaModel
from IPython.core.display import HTML
model_dir = 'temp'
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = RobertaForSequenceClassification.from_pretrained(model_dir)
tokenizer1 = RobertaTokenizer.from_pretrained('roberta-base')
model1 = RobertaModel.from_pretrained('roberta-base')
threshold=0.5
def process_text(input_text):
    if input_text:
        text = input_text
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1)
    max_prob, predicted_class_id = torch.max(probs, dim=1)
    prob = str(round(max_prob.item() * 100, 2))
    label = model.config.id2label[predicted_class_id.item()]
    final_label='Human' if model.config.id2label[predicted_class_id.item()]=='LABEL_0' else 'Chat-GPT'
    processed_result = text
    def search(text):
        query = text
        api_key = 'AIzaSyClvkiiJTZrCJ8BLqUY9I38WYmbve8g-c8'
        search_engine_id = '53d064810efa44ce7'
        url = f'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}&num=5'

        try:
            response = requests.get(url)
            data = response.json()
            return data
        except Exception as e:
            return {'error': str(e)}
    def get_article_text(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text from the article content (you may need to adjust this based on the website's structure)
                article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
            return article_text
        except Exception as e:
            print(f"An error occurred: {e}")
        return ''
    def find_plagiarism(text):
        search_results=[]
        if len(text)>300:
            search_results = search(text)
        if 'items' not in search_results:
            return []
        similar_articles = []
        for item in search_results['items']:
            link = item.get('link', '')
            article_text = get_article_text(link)
            if article_text:
            # Tokenize and encode the input text and the article text
                encoding1 = tokenizer1(text, max_length=512, truncation=True, padding=True, return_tensors="pt")
                encoding2 = tokenizer1(article_text, max_length=512, truncation=True, padding=True, return_tensors="pt")
            
            # Calculate embeddings using the model
                with torch.no_grad():
                    embedding1 = model1(**encoding1).last_hidden_state.mean(dim=1)
                    embedding2 = model1(**encoding2).last_hidden_state.mean(dim=1)
            
            # Calculate cosine similarity between the input text and the article text embeddings
                similarity = cosine_similarity(embedding1, embedding2)[0][0]
                if similarity > threshold:
                    similar_articles.append([link,float(similarity)])
        similar_articles = sorted(similar_articles, key=lambda x: x[1], reverse=True)
        return similar_articles[:5]

    # prediction = pipe([text])
    # explainer = shap.DeepExplainer(model,[text])
    # shap_values = explainer([text])
    # shap_plot_html = HTML(shap.plots.text(shap_values, display=False)).data
    similar_articles = find_plagiarism(text)

    return processed_result, prob, final_label,similar_articles

text_input = gr.Textbox(label="Enter text")
outputs = [gr.Textbox(label="Processed text"), gr.Textbox(label="Probability"), gr.Textbox(label="Label"),gr.Dataframe(label="Similar Articles", headers=["Link", "Similarity"],row_count=5)]
title = "Group 2- ChatGPT text detection module"
description = '''Please upload text files and text input responsibly and await the explainable results. The approach in place includes finetuning a Roberta model for text classification.Once the classifications are done the most similar articles are presented along with the alleged similarity'''
gr.Interface(fn=process_text,title=title,description=description, inputs=[text_input], outputs=outputs).launch()

