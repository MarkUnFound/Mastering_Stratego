import re

with open(r'c:\Users\Mark Lawrence Quibot\repo\Research\Latex\references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

entries = re.findall(r'@\w+\{([^,]+),([\s\S]*?)(?=\n@|\Z)', text)
seen = {}

for key, content in entries:
    title_match = re.search(r'title\s*=\s*(?:\{|")([^}]+)(?:\}|")', content, re.IGNORECASE | re.DOTALL)
    if not title_match: continue
    title = title_match.group(1).replace('\n', ' ')
    title = re.sub(r'\s+', ' ', title)
    title_clean = re.sub(r'[^a-zA-Z0-9]', '', title).lower()
    
    if title_clean in seen:
        print('Duplicate:', key, 'and', seen[title_clean])
    else:
        seen[title_clean] = key
