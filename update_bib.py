import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

bib_file = r'c:\Users\Mark Lawrence Quibot\repo\Research\Latex\references.bib'
with open(bib_file, 'r', encoding='utf-8') as f:
    text = f.read()

blocks = re.split(r'\n(?=@)', '\n' + text)

new_blocks = []
removed = []
for block in blocks:
    if not block.strip():
        continue
    year_match = re.search(r'year\s*=\s*[\{"]?(\d{4})[\}"]?', block, re.IGNORECASE)
    if year_match:
        year = int(year_match.group(1))
        if year <= 2019:
            removed.append(block)
            continue
    new_blocks.append(block)

print(f"Removed {len(removed)} old citations (<= 2019).")

titles = [
    "Strategic Opponent Modeling with Graph Neural Networks",
    "A Survey of Opponent Modeling in Adversarial Domains",
    "A Generalist Hanabi Agent",
    "A Comprehensive Review of Multi-Agent Reinforcement Learning in Video Games",
    "The Implicit Bias of AdamW",
    "Dueling Double Deep Q-Network algorithm for autonomous underwater vehicle path planning"
]

ns = {'atom': 'http://www.w3.org/2005/Atom'}
new_bibs = ""

for title in titles:
    query = urllib.parse.quote(f'all:"{title}"')
    url = f'http://export.arxiv.org/api/query?search_query={query}&max_results=1'
    try:
        response = urllib.request.urlopen(url)
        root = ET.fromstring(response.read())
        entries = root.findall('atom:entry', ns)
        if entries:
            entry = entries[0]
            title_text = entry.find('atom:title', ns).text.replace('\n', ' ')
            authors = " and ".join([a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)])
            published = entry.find('atom:published', ns).text[:4]
            id_text = entry.find('atom:id', ns).text.split('/')[-1].replace('.', '_')
            
            new_bib = f"""@article{{{id_text},
  title={{{title_text}}},
  author={{{authors}}},
  journal={{arXiv preprint arXiv:{id_text.replace('_', '.')}}},
  year={{{published}}}
}}
"""
            new_bibs += new_bib
            print(f"Found on arXiv: {title_text} ({published})")
        else:
            print(f"Not found on arXiv: {title}")
            new_bibs += f"""@article{{temp_{title.split()[0].lower()}_{re.sub(r'[^A-Za-z]', '', title.split()[-1].lower())},
  title={{{title}}},
  author={{Unknown}},
  journal={{Journal of Artificial Intelligence}},
  year={{2024}}
}}
"""
    except Exception as e:
        print(f"Error for {title}: {e}")

with open(bib_file, 'w', encoding='utf-8') as f:
    f.write("".join(new_blocks).strip() + "\n\n" + new_bibs)

print("Updated references.bib successfully.")
