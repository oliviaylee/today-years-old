import json

# Generating JSON Dataset
def generate_json_subset(n=50000):
    with open("datasets/urban_words.json", 'r') as f:
        with open("datasets/urban_words_50000.json", 'a') as g:
            counter = 0
            for line in f.readlines():
                g.write(line)
                counter += 1
            if counter == n: return

# Check enough novel tokens
def novel_tokens_check():
    counter = 0
    for line in open('urban_words.json', "r"):
        if counter == 10000:
            print("done")
            break
        entry = json.loads(line)
        print(entry)
        input()
        word, defn, upv, downv = entry['lowercase_word'], entry['definition'].lower(), int(entry["thumbs_up"]), int(entry["thumbs_down"])
        if (len(word.split(' ')) > 1) or (downv > upv) or (upv < 10): continue
        counter += 1