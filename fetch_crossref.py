import urllib.request
import json
import urllib.parse
titles = [
    "Strategic Opponent Modeling with Graph Neural Networks",
    "A Survey of Opponent Modeling in Adversarial Domains",
    "The Implicit Bias of AdamW: Constrained Optimization",
    "Dueling Double Deep Q-Network algorithm for autonomous underwater vehicle path planning"
]
for title in titles:
    query = urllib.parse.quote(title)
    url = f'https://api.crossref.org/works?query.title={query}&rows=1'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'mailto:test@example.com'})
        res = urllib.request.urlopen(req)
        data = json.loads(res.read())
        if data['message']['items']:
            item = data['message']['items'][0]
            authors = " and ".join([a.get('given', '') + ' ' + a.get('family', '') for a in item.get('author', [])])
            year = item.get('issued', {}).get('date-parts', [[None]])[0][0]
            print(f"Title: {item.get('title', [''])[0]}"[:60])
            print(f"Authors: {authors}")
            print(f"Year: {year}\n")
    except Exception as e:
        pass
