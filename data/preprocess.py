import re
from pathlib import Path

def preprocess(filename, raw_path, output_path):
    lines = []
    with open(raw_path/filename, "r") as f:
        for line in f.readlines():
            line = line.replace('[', '').replace(']', '')
            line = remove_id(line)
            lines.append(line)
    with open(output_path/filename, "w") as f:
        for line in lines:
            f.write(line) 

def remove_id(sentence):
    pattern = r'^(?:\d+\s)+'
    result = re.sub(pattern, '', sentence)
    return result   

if __name__ == "__main__":
    raw_path = Path("winobias/templates")
    output_path = Path("winobias")
    for f in raw_path.glob("*.*"):
        print(f"Preprocessing {f.name}")
        preprocess(f.name ,raw_path=raw_path, output_path=output_path)
