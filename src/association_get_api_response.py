from openai import OpenAI
import tqdm
import pandas as pd
from queue import Queue
import json
import json_repair
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_second_three_words(seed, model, temperature, max_retries=3):
    message = [{"role": "user",
                "content": f"""Starting with the word "{seed}", generate three different words that directly associate with this initial word only (not with each other). Please put down only single words, and do not use proper nouns (such as names, brands, etc.). For each word, provide a brief explanation of its connection to "{seed}". Return in JSON format {{"results":[{{"word":"","reason":""}},{{"word":"","reason":""}},{{"word":"","reason":""}}]}}."""}]
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model = model,
                messages = message,
                temperature = temperature
            )
            content = json_repair.loads(response.choices[0].message.content)
            return content
        except Exception as e:
            retries += 1
            if retries == max_retries:
                raise e
    return None

def get_third_association_chains(seed, second_word, second_word_reason, model, temperature, max_retries=5):
    message = [{"role": "user",
                "content": f"""Starting with the word pair "{seed}" → "{second_word}", generate a chain of 20 words where each new word should be associated with ONLY the word immediately before it. Generate the third word based on "{second_word}", then generate the fourth word based on your third word, and so on. Please put down only single words, and do not use proper nouns (such as names, brands, etc.). For each word, provide a brief explanation of its connection to the previous word. Return in JSON format with exactly 20 entries {{"results":[{{"word":"{second_word}","reason":"{second_word_reason}"}},{{"word":"","reason":""}},{{"word":"","reason":""}}...{{"word":"","reason":""}}]}}."""}]
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model = model,
                messages = message,
                temperature = temperature
            )
            content = json_repair.loads(response.choices[0].message.content)
            if len(content['results']) >= 19:
                return content
            else:
                retries += 1
                continue
        except Exception as e:
            retries += 1
            if retries == max_retries:
                raise e
    return None

def process_entry(entry, model, temperature):
    seed = entry['seed']
    # generate three words based on <seed>
    response_second_word = get_second_three_words(seed, model, temperature)

    # generate three chains based on each <second word>
    chain_output_dict = {}
    if response_second_word:
        index = ['first_response', 'second_response', 'third_response']
        for i, second_word_info in enumerate(response_second_word.get('results', [])):
            second_word = second_word_info['word']
            second_word_reason = second_word_info['reason']
            chain = get_third_association_chains(seed,second_word,second_word_reason,model,temperature)
            chain_output_dict[index[i]] = chain.get('results') if chain else []

    output = {
        'input': {'model': model, 'temperature': temperature},
        'output': chain_output_dict,
        'cue_word_info': entry
    }
    print(output)
    return output

if __name__ == '__main__':

    association_word_file_path = r'../data/cue_words_ids_with_coca_n.xlsx'

    df = pd.read_excel(association_word_file_path)
    seeds = df.to_dict(orient='records')

    for i,seed in enumerate(seeds):
        seed['index'] = i+1

    result_queue = Queue()

    api_key = ''
    api_base = ''
    model = ''

    client = OpenAI(
        api_key = api_key,
        base_url= api_base
    )

    if '/' in model:
        model_name = model.split('/')[1]  # openrouter
        if ':' in model:
            model_name = model_name.split(':')[0]
        output_file = rf'./response/association_response_{model_name}.jsonl'
    else:
        output_file = rf'./response/association_response_{model}.jsonl'

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_entry, entry, model, temperature=0) for entry in seeds]

        count = 0
        batch_size = 10
        batch_results = []

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            batch_results.append(result)
            count += 1

            if count % batch_size == 0:
                batch_results.sort(key=lambda x: x['cue_word_info']['index'])
                with open(output_file, 'a', encoding='utf-8') as f:
                    for output in batch_results:
                        f.write(json.dumps(output, ensure_ascii=False) + '\n')
                batch_results = []

        if batch_results:
            batch_results.sort(key=lambda x: x['cue_word_info']['index'])
            with open(output_file, 'a', encoding='utf-8') as f:
                for output in batch_results:
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')

