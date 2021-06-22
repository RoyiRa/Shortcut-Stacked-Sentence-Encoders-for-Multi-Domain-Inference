import jsonlines

# with open('snli_1.0/snli_1.0_train.jsonl') as json_file:
#     data = json.load(json_file)
#     # print(data[0])

objs = []
with jsonlines.open('snli_1.0/snli_1.0_train.jsonl') as reader:
    for obj in reader:
        objs.append(obj)
print(objs[0]['sentence1'])
print(objs[0]['sentence2'])
print(objs[0])