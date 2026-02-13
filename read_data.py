import json
pprint = lambda x: print(json.dumps(x, indent=2)) if isinstance(x, dict) else display(x)

#file = "meta_Pet_Supplies.jsonl/meta_Pet_Supplies.jsonl"
file = "Pet_Supplies.jsonl/Pet_Supplies.jsonl"

with open(file, 'r') as fp:
    for line in fp:
        pprint(json.loads(line.strip()))
        break


# file = # e.g., "All_Beauty.jsonl", downloaded from the `review` link above
# with open(file, 'r') as fp:
#     for line in fp:
#         print(json.loads(line.strip()))


# file = # e.g., "meta_All_Beauty.jsonl", downloaded from the `meta` link above
# with open(file, 'r') as fp:
#     for line in fp:
#         print(json.loads(line.strip()))

"""
file = "meta_categories/meta_All_Beauty.jsonl"
with open(file, 'r') as fp:
    for line in fp:
        pprint(json.loads(line.strip()))
        break
"""
