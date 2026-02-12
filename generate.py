from transformers import pipeline
from transformers import AutoTokenizer
from datetime import datetime
import torch
import json
import os

MODEL_NAME = "SemanticAlignment/Mistral-v0.1-Italian-LAPT-instruct"


### PARAMETERS
lang = "it"

genre = "Biography"

prefix = "Minerva7B_ItBio"

temperatures = [0.7, 1.0, 1.3]

reiterations = 25

personalities = ["Dacia Maraini", "Gae Aulenti"]

SYSTEM_PROMPT = "Il tuo compito è generare testi informativi."

USER_PROMPT = "Scrivi una biografia su {person}."

output_folder = "outputs"

# output name: Minerva7B_ItBio_07_Dacia_Maraini_001

def main():
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16
    )

    conversations = []

    output_dir = os.path.join(output_folder, prefix)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for temperature in temperatures:
        for personality in personalities:
        
            personality_prompt = USER_PROMPT.format(person=personality)
            
            conversations.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": personality_prompt},
            ])

            chat_samples = tokenizer.apply_chat_template(conversations, tokenize=False)

            # get number of prompt tokens
            prompt_tokens_number = len(tokenizer(chat_samples)["input_ids"])

            outputs = generator(
                conversations,
                max_new_tokens=2048,
                temperature=temperature,
                num_return_sequences=25,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # 👈 critical
                ],
            )

            for i, out in enumerate(outputs[0]):
                base_output_name = f"{prefix}_{temperature}_{personality}_{i}".replace(".", "").replace(" ", "_")

                generated_text = out["generated_text"][-1]["content"]

                print(generated_text)

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
                        "model": MODEL_NAME,
                        "temperature": temperature,
                        "date": datetime.today().strftime('%Y-%m-%d'),
                        "token_usage": {
                            "prompt_tokens": prompt_tokens_number,
                            "completion_tokens": completions_tokens_number,
                            "total_tokens": prompt_tokens_number + completions_tokens_number
                        }
                    },
                    f_json,
                    indent=4)

                input()


if __name__ == "__main__":
    main()