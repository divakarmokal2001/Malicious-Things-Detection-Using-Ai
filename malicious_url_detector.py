import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertModel
from urllib.parse import urlparse
import re

def extract_features(url, html_content):
    parsed_url = urlparse(url)
    path = parsed_url.path
    query = parsed_url.query
    fragment = parsed_url.fragment
    url_features = path + query + fragment

    # Extract features from HTML content
    script_tags = re.findall(r'<script.*?>.*?</script>', html_content, flags=re.DOTALL)
    script_content = ' '.join(script_tags)

    # New feature: BERT embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(url_features, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.pooler_output.detach().numpy().ravel()
    
    return url_features + ' ' + script_content + ' ' + ' '.join(map(str, embeddings))

def is_malicious(url):
    try:
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Error fetching URL content: {e}")
        return False

    features = extract_features(url, html_content)

    # Training a TF-IDF vectorizer and a Naive Bayes classifier
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([features])
    clf = MultinomialNB()
    clf.fit(X, [0])  # Assuming 0 represents non-malicious and 1 represents malicious

    # Predicting whether the URL is malicious
    prediction = clf.predict(X)

    if bool(prediction[0]):
        print(f"The URL {url} is malicious.")
        print("Features:")
        print(f"- URL Path: {parsed_url.path}")
        print(f"- URL Query: {parsed_url.query}")
        print(f"- URL Fragment: {parsed_url.fragment}")
        print(f"- Script Content: {script_content}")
        print(f"- BERT Embeddings: {embeddings}")
    else:
        print(f"The URL {url} is not malicious.")

if __name__ == "__main__":
    # Example usage
    url_to_check = "http://example.com"
    is_malicious(url_to_check)

