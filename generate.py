from transformers import pipeline
from transformers import AutoTokenizer
from datetime import datetime
import torch
import json
import os

import argparse


### CONSTANTS
SYSTEM_PROMPT = "Il tuo compito è generare testi informativi."

USER_PROMPT = {
    "Biography": "Scrivi una biografia su {x}",
    "News_Climate": "Scrivi un testo giornalistico sul cambiamento climatico a {x}",
    "Fable": "Scrivi una favola per bambini {x}",
    "Press_Release": "Scrivi un comunicato stampa sul modello linguistico di grandi dimensioni {x}",
    "Political_Speech": "Scrivi un discorso politico sul cambiamento climatico dalla prospettiva di un partito {x}"
}

def main(args):
    model_name = args.model_name
    language = args.language
    genre = args.genre
    temperatures = args.temperatures
    reiterations = args.reiterations
    prompt_params = args.prompt_params
    output_folder = args.output_folder

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        dtype=torch.bfloat16
    )

    conversations = []

    model_prefix = model_name.split("/")[-1]

    output_dir = os.path.join(output_folder, model_prefix, genre)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating model: {model_prefix}")

    for temperature in temperatures:
        for prompt_param in prompt_params:
            personality_prompt = USER_PROMPT[genre].format(x=prompt_param)
            print(f"\tGenerating text with temperature: {temperature} | and parameter: {prompt_param}\n\t\tRunning Prompt: {personality_prompt}")

            conversations = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": personality_prompt},
            ]

            chat_samples = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

            # get number of prompt tokens
            prompt_tokens_number = len(tokenizer(chat_samples)["input_ids"])

            outputs = generator(
                conversations,
                max_new_tokens=2048,
                temperature=temperature,
                num_return_sequences=int(reiterations),
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # 👈 critical
                ],
            )

            for i, out in enumerate(outputs):
                base_output_name = f"{model_prefix}_{temperature}_{prompt_param}_{i}".replace(".", "").replace(" ", "_")

                generated_text = out["generated_text"][-1]["content"]

                completions_tokens_number = len(tokenizer(generated_text)["input_ids"])

                json_output_name = base_output_name+".json"
                txt_output_name = base_output_name+".txt"

                with open(os.path.join(output_dir, json_output_name), "w") as f_json, open(os.path.join(output_dir, txt_output_name), "w") as f_txt:
                    f_txt.writelines(generated_text)
                    json.dump({
                        "language": language,
                        "genre": genre,
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": personality_prompt,
                        "model": model_prefix,
                        "temperature": temperature,
                        "date": datetime.today().strftime('%Y-%m-%d'),
                        "text": generated_text,
                        "token_usage": {
                            "prompt_tokens": prompt_tokens_number,
                            "completion_tokens": completions_tokens_number,
                            "total_tokens": prompt_tokens_number + completions_tokens_number
                        }
                    },
                    f_json,
                    indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    # CLI Parameters
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-l', '--language')
    parser.add_argument('-g', '--genre')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-t', '--temperatures', type=float, nargs="+")
    parser.add_argument('-r', '--reiterations')
    parser.add_argument('-e', '--prompt_params', nargs="+")
    parser.add_argument('-o', '--output_folder')

    args = parser.parse_args()

    main(args)