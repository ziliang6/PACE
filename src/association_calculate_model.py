import json
import numpy as np
from scipy.spatial.distance import cosine
from concurrent.futures import ProcessPoolExecutor
import os
import shutil
import json_repair

word_vector_path = '' # e.g., './models/glove.6B.300d.txt'

word_vector = {}
with open(word_vector_path, 'r', encoding='utf-8')as f:
    for line in f:
        splitline = line.rstrip().split(' ')
        word = splitline[0]
        embedding = np.asarray(splitline[1:], dtype='float32')
        word_vector[word] = embedding

mean_embedding = np.squeeze(np.mean(list(word_vector.values()), axis=0))

def get_word_embedding(word):
    word = word.lower()  # glove 42b uncased
    if word in word_vector:
        return np.squeeze(word_vector[word])
    else:
        print(f'None Embedding:{word}')
        return mean_embedding

def cosine_similarity_distance(word1, word2):
    vec1 = get_word_embedding(word1)
    vec2 = get_word_embedding(word2)
    cosine_sim = cosine(vec1, vec2)
    return cosine_sim

def generate_output_filename(input_filename):
    prefix = 'association_response_'
    start_index = input_filename.find(prefix) + len(prefix)
    end_index = input_filename.rfind('.')
    extracted_info = input_filename[start_index:end_index]
    calculate_file_path = f'./calculation/association_calculate_{extracted_info}.jsonl'
    return calculate_file_path

def calculate_ttr(response):
    num_tokens = len(response)
    num_types = len(set(response))
    ttr = num_types / num_tokens
    return ttr

def calculate(response):
    cur = []
    for i in range(1,len(response)):
        tmp = 0
        for j in range(i):
            tmp += cosine_similarity_distance(response[i],response[j])
        cur.append(tmp / i)
    association_distance = sum(cur)/len(cur)
    return association_distance

def processing_line(line):
    line = json_repair.loads(line)
    output = line['output']
    seed = line['cue_word_info']['seed']

    association_distances = []
    all_layer_word_response = []

    for _,layer_full_response in output.items():
        layer_word_response = [seed]
        for item in layer_full_response:
            if isinstance(item, dict) and 'word' in item:
                layer_word_response.append(item['word'])
            else:
                print(f"Warning: Invalid item format: {item}")
                continue

        if len(layer_word_response) < 20:
            print(f"Warning: Response {layer_word_response} is less than 20")

        if layer_word_response[1].lower() == seed.lower(): # sometimes the model regenerates the seed word in its response
            layer_word_response.remove(layer_word_response[1])
            
        layer_word_response = layer_word_response[:20] # some models may generate chains longer than the expected length
        all_layer_word_response.extend(layer_word_response[1:]) # this avoids duplicate computation of the seed in association distance calculation
        association_distance = calculate(layer_word_response)

        association_distances.append(association_distance)

    ttr = calculate_ttr(all_layer_word_response)

    line['calculation'] = {'ttr':ttr, 'association_distance':np.mean(association_distances)} # Average over 3 chains
    return json.dumps(line, ensure_ascii=False) + '\n' 

def main(input_file_path, num_workers):
    print('start:',input_file_path)
    calculate_file_path = generate_output_filename(input_file_path)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(processing_line, lines))
    with open(calculate_file_path, 'w', encoding='utf-8') as out_file:
        out_file.writelines(results)

    print('done:',input_file_path)

if __name__ == "__main__":
    num_workers = 1
    directory = r'../data/response'
    files = [os.path.join(directory, filename)for filename in os.listdir(directory)]

    folder_path = './calculation'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    for file in files:
        main(file,num_workers)
