import json
def check(f):
    data = json.load(open(f, 'r', encoding='utf-8'))
    for k, v in list(data.items()):
        if isinstance(v, list):
            print(f"  {k}: list of len {len(v)}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
            
print("LSTMDQN")
check(r"c:\Users\Mark Lawrence Quibot\repo\Research\History\LSTMDQN RUN\training_history.json")
print("Rainbow DQN")
check(r"c:\Users\Mark Lawrence Quibot\repo\Research\Modular Stratego\dqn_models\training_history.json")
