import json

# counter = 0
# with open("datasets/urban_words.json", 'r') as f:
#     with open("datasets/urban_words_50000", 'a') as g:
#         for line in f.readlines():
#             if counter > 50000: break
#             g.write(line)
#             counter += 1

counter = 0
for line in open('urban_words_50000.json', "r"):
    if counter == 10000:
        print("done")
        break
    entry = json.loads(line)
    word, defn, upv, downv = entry['lowercase_word'], entry['definition'].lower(), int(entry["thumbs_up"]), int(entry["thumbs_down"])
    if (len(word.split(' ')) > 1) or (downv > upv) or (upv < 10): continue
    counter += 1