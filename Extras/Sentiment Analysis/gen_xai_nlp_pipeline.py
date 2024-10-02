import torch
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch     
from torch import nn      
import torch.nn.functional as F
from peft import LoraConfig
from trl import DPOTrainer
from accelerate import Accelerator
from datasets import Dataset
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
import random
import string
import spacy
from d2l import torch as d2l

from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
from captum.concept import TCAV
from captum.concept import Concept
from torch.utils.data import DataLoader, TensorDataset

import os
# Download the stopwords set
nltk.download('stopwords')
import shutil
# RL Imports
import gym
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import torch
from transformers import CLIPProcessor, CLIPModel
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.") 


# List of positive and negative prompts define by user.
positive_prompts = [
    "The customer service team was very helpful and responsive when I reached out for support. They were patient and provided clear instructions on how to address some of the issues, which improved the situation slightly.",
    "The customer support was excellent and helped me resolve some problems quickly.",
    "The support team was prompt and provided useful troubleshooting tips.",
    "The customer service was very understanding and guided me through some fixes.",
    "The customer service was very responsive and sent a replacement quickly.",
    "The support team was very helpful and offered solutions that partially improved the experience.",
    "The customer service was very efficient and arranged a replacement promptly.",
    "The customer support was excellent, providing detailed instructions to fix the issues.",
    "The customer service was superb, offering a refund without any hassle.",
    "The support team was very friendly and helped arrange a refund.",
    "The customer service was quick to respond and arranged for a free repair."
]

negative_prompts = [
"The highly anticipated movie turned out to be a colossal disappointment, plagued by a weak and incoherent plot, unconvincing performances by the lead actors, lackluster special effects, and numerous continuity errors, which collectively made it one of the worst cinematic experiences in recent memory, leaving audiences and critics alike utterly dissatisfied and frustrated.",
    "I bought a new smartphone hoping for great features and a long battery life, but it often overheats and the screen glitches.",
    "My recent purchase of a high-end camera was underwhelming. The picture quality is not as sharp as advertised, and it freezes during use.",
    "I was excited about my new tablet, expecting seamless performance, but it lags and apps crash frequently.",
    "After purchasing a premium headset, I was disappointed by the poor sound quality and uncomfortable fit.",
    "I invested in a top-tier smart TV, but it has connectivity issues and often restarts itself.",
    "I recently bought an expensive blender that stopped working after a few uses.",
    "My new high-tech washing machine fails to complete cycles and leaves clothes damp.",
    "The latest gaming console I purchased crashes during gameplay and has overheating problems.",
    "I ordered a high-end coffee maker that leaks and doesn't brew properly.",
    "The luxury watch I bought stopped working within a month, and the strap feels cheap."
]

# Generate random gibberish words and sentences
def generate_gibberish_word():
    """Generate a gibberish word of random length between 2 and 7."""
    length = random.randint(2, 7)
    letters = string.ascii_lowercase  # use lowercase letters
    return ''.join(random.choice(letters) for _ in range(length))

def generate_gibberish_sentence(min_words=2, max_words=5):
    """Generate a gibberish sentence with a random number of words between min_words and max_words."""
    num_words = random.randint(min_words, max_words)
    return ' '.join(generate_gibberish_word() for _ in range(num_words))

def generate_gibberish_list(size, min_words=5, max_words=5):
    """Generate a list of gibberish sentences of random lengths."""
    return [generate_gibberish_sentence(min_words, max_words) for _ in range(size)]

def repeat_words(input_list, times=5):
    """Convert a list of words to a list where each word is repeated a specified number of times."""
    return [f"{' '.join([word] * times)}" for word in input_list]


def assemble_token(data_list, id, name):

    new_list = [torch.tensor([vocab[i.split()]], device='cpu').reshape(1, -1) for i in data_list]
    # print(new_list)
    # Use the TensorDataset to hold the reshaped tensors
    dataset = TensorDataset(torch.cat(new_list, dim=0))  # Concatenate along the sequence length dimension
    
    # DataLoader to yield one item at a time
    concept_iter = DataLoader(dataset, batch_size=1, shuffle=False)
    
    return Concept(id=id, name=name, data_iter=concept_iter)


# Generation of concepts using Mistral
def generate_concept(word, output_path="", use_dpo=False):
    os.makedirs(output_path, exist_ok=True)

    input_prompt = [
        {"role": "user", "content": "What are the synonyms of angry?"},
        {"role": "assistant", "content": "[furious,mad,pissed,enraged,rage,hastle]"},
        {"role": "user", "content": f"What are the synonyms of {word}?. Only give the synonym word as outputs seperated by ,"},
    ]

    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2' if not use_dpo else output_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2' if not use_dpo else output_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Input prompt
    if not isinstance(input_prompt, list):
        prompt = [input_prompt]
    else:
        prompt = input_prompt
    input_ids = tokenizer.apply_chat_template(prompt, padding=True, return_tensors="pt").to(model.device)

    # Generate concept
    output = model.generate(input_ids, max_new_tokens=20, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    # Decode output
    decoded_output = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Free memory
    del model, tokenizer
    Accelerator().free_memory()
    torch.cuda.empty_cache()

    final_output = cleaned_list = list(filter(None, decoded_output.split(','))) 
    cleaned_final_list = [string.strip() for string in final_output]
    
    return cleaned_final_list


def apply_dpo(dpo_preferred_concepts, dpo_unpreferred_concepts, prompt_word, num_epochs=2, continue_training=False, output_path=""):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + '/checkpoints', exist_ok=True)

    torch_dtype = torch.float16

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        max_steps=0,
        save_steps=2,
        # gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=5e-4,
        output_dir=f"{output_path}/checkpoints",
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        run_name="dpo_causal_lm",
        remove_unused_columns=False,
    )

    input_prompt = '''
        {"role": "user", "content": "What are the synonyms of angry?"},
        {"role": "assistant", "content": "[furious,mad,pissed,enraged,rage,hastle]"},
        {"role": "user", "content": f"What are the synonyms of''' + prompt_word + '''? Only give the synonym word as outputs seperated by ,"},
    '''

    prompts = [input_prompt] * len(dpo_preferred_concepts)

    dpo_dataset_dict = {
        "prompt": prompts,
        "chosen": dpo_preferred_concepts,
        "rejected": dpo_unpreferred_concepts,
    }

    train_dataset = Dataset.from_dict(dpo_dataset_dict)
    train_dataset = train_dataset.map(
        batched=True,
        num_proc=1,
    )

    # Update the NLP model here
    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2' if not continue_training else output_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2' if not continue_training else output_path)
    tokenizer.pad_token = tokenizer.eos_token

    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=128,
        max_length=60,
    )
    
    dpo_trainer.args.max_steps = num_epochs

    if continue_training:
        dpo_trainer.train(resume_from_checkpoint=continue_training)
    else:
        dpo_trainer.train()

    dpo_trainer.save_model(output_path)

    # Free memory
    del model, tokenizer, dpo_trainer
    Accelerator().free_memory()
    torch.cuda.empty_cache()


# Define positive or negative prompt
prompt = negative_prompts[0]
key = prompt.split(" ")
keywords = []
for i in key:
    if len(i) != 1:
        keywords.append(i)

stop_words = set(stopwords.words('english'))

# Filter out stop words from your list
keywords = [word for word in keywords if word not in stop_words]

print(f"length of words in prompt : {len(keywords)}")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

similarity_threshold = 0.95

words = keywords

all_embs = []
with torch.no_grad():
    for word in words:
        inputs = clip_processor(text=word, return_tensors="pt", padding=True).to(device)
        outputs = clip_model.get_text_features(**inputs)
        all_embs.append(outputs.squeeze(0))

all_embs = torch.stack(all_embs)
all_embs = all_embs / all_embs.norm(dim=1, keepdim=True)

similarity_matrix = all_embs @ all_embs.T
similarity_matrix = similarity_matrix.cpu().numpy()

filtered_keywords = []
duplicate = set()
for i in range(len(words)):
    if i in duplicate:
        continue
    filtered_keywords.append(words[i])
    for j in range(i+1, len(words)):
        if similarity_matrix[i][j] > similarity_threshold:
            duplicate.add(j)

print(f"length of filtered words in prompt : {len(filtered_keywords)}")


# Reinforcement
# Getting the dataset for sentiment model: 
batch_size = 64                                                  
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)    

#  Sentiment Model 
class TextCNN(nn.Module):                                                     
  def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, 
               **kwargs):                                                      
    super(TextCNN, self).__init__(**kwargs)
    self.embedding = nn.Embedding(vocab_size, embed_size)                     
    self.constant_embedding = nn.Embedding(vocab_size, embed_size)           
    self.dropout = nn.Dropout(0.5)                                             
    self.decoder = nn.Linear(sum(num_channels), 2)                            
    self.pool = nn.AdaptiveAvgPool1d(1)                                        
    self.relu = nn.ReLU()                                                     
    self.convs = nn.ModuleList()
    for c, k in zip(num_channels, kernel_sizes):
      self.convs.append(nn.Conv1d(2 * embed_size, c, k))                     
  
  def forward(self, inputs):
    embeddings = torch.cat((self.embedding(inputs), 
                            self.constant_embedding(inputs)), dim=2)          
    embeddings = embeddings.permute(0, 2, 1)                                   
    encoding = torch.cat([torch.squeeze(
        self.relu(self.pool(conv(embeddings))), dim=-1) for 
        conv in self.convs], dim=1)                                           
    outputs = self.decoder(self.dropout(encoding))                            
    return outputs
 
embed_size, kernel_sizes, num_channels = 100, [3, 4, 5], [100, 100, 100]       
devices = d2l.try_all_gpus()                                                  
net = TextCNN(len(vocab), embed_size, kernel_sizes, num_channels)             

net = torch.load('weights/model_sentiment.pth')
net.eval()


def get_reward(scores, layers, concepts):
    keys = [key for key in scores.keys()]
    # rewards = {}
    # for i in range(len(concepts)):
    #     value = [format_float(scores['sign_count'][i])] # ... Incomplete

    reward = scores[keys[0]][layers[0]]['sign_count'][0] 
    return reward

def remove_cav_folder():
    if os.path.exists('cav'):
        shutil.rmtree('cav')


class GenConceptEnv(gym.Env):
    def __init__(self, filtered_keywords):
        super(GenConceptEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(len(filtered_keywords))
        self.actions = filtered_keywords
        self.action_list = []
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 512), dtype=np.float32)
        self.prompt = 'prompt'
        self.prompt_em = ''
        self.best_action_score = [-1.0] * len(self.actions)

        self.use_dpo_weights = False
        self.dpo_epochs = 2

    def reset(self):

        remove_cav_folder()

        prompt_index = 0
        # Define positive or negative prompt
        self.prompt = negative_prompts[prompt_index]

        clip_input = clip_processor(text=self.prompt, return_tensors="pt", padding=True).to(device)
        clip_embed = clip_model.get_text_features(**clip_input)
        
        obs = clip_embed
        return obs.cpu().detach().numpy()

    def step(self, action):

        current_concept= self.actions[action]
        
        self.action_list.append(current_concept)

        count_action = self.action_list.count(current_concept)
        alpha = 2*count_action/100

        temp_concepts = generate_concept(current_concept, use_dpo=self.use_dpo_weights, output_path='dpo test')
        concepts = []
        for i in temp_concepts:
            concepts.append(i.split()[0])

        midpoint = len(concepts) // 2

        if len(concepts) % 2 != 0:
            midpoint += 1

        p1 = concepts[:midpoint]
        p2 = concepts[midpoint:]
        # Split the list into two parts
        c1 = repeat_words(concepts[:midpoint], 5)
        c2 = repeat_words(concepts[midpoint:], 5)

        # Ensure both halves have the same size
        if len(c1) > len(c2):
            c1.pop()
            p1.pop()
        elif len(c1) > len(c2):
            c2.pop()
            p2.pop()

        random_concepts = generate_gibberish_list(len(c1))


        tcav_obj = TCAV(
            model=net,
            layers=["decoder"],
            # classifier=classifier,
            layer_attr_method = LayerIntegratedGradients(
                net,
                None,
                multiply_by_inputs=False
            )
        )

        if len(c1) == 0:
            pref_clip_input = clip_processor(text=str(p1), return_tensors="pt", padding=True).to(device)
            obs = clip_model.get_text_features(**pref_clip_input)
            return None, 0, True, {}
        
        if len(c1) == 1:
            c1.append(c1[0])
            c2.append(c2[0])
            random_concepts = generate_gibberish_list(len(c1))

        random_c = assemble_token(name='random_concept', id=0, data_list=random_concepts)
        gen_c1 = assemble_token(name='c1', id=1, data_list=c1)
        gen_c2 = assemble_token(name='c2', id=2, data_list=c2)

        exp_set_1 = [[gen_c1, random_c]]
        exp_set_2 = [[gen_c2, random_c]]

        tcav_score_1 = tcav_obj.interpret(
            inputs=torch.tensor(vocab[self.prompt.split()], device='cpu').reshape(1, -1),
            experimental_sets=exp_set_1,
            target=1,
            n_steps=5,
        )
        # print(tcav_score_1)
        tcav_score_2 = tcav_obj.interpret(
            inputs=torch.tensor(vocab[self.prompt.split()], device='cpu').reshape(1, -1),
            experimental_sets=exp_set_2,
            target=0,
            n_steps=5,
        )
        score1 = get_reward(tcav_score_1, ['decoder'], exp_set_1)
        score2 = get_reward(tcav_score_2, ['decoder'], exp_set_2)

        both_positive_and_high = False
        both_negative_and_low = False

        if score1 > 0.7 and score2 > 0.7:
            alpha = 1
            both_positive_and_high = True

        if score1 < 0 and score2 < 0:
            both_negative_and_low = True

        if score1 > score2:
            best_score = score1
            preferred_concepts = p1
            unpreferred_concepts = p2
            if score1 >= self.best_action_score[action]:
                self.best_action_score[action] = score1
        else:
            best_score = score2
            preferred_concepts = p2
            unpreferred_concepts = p1
            if score2 >= self.best_action_score[action]:
                self.best_action_score[action] = score2
        
        if (best_score == self.best_action_score[action]) and not both_positive_and_high:
            apply_dpo(preferred_concepts, unpreferred_concepts, current_concept, num_epochs=self.dpo_epochs, continue_training=self.use_dpo_weights, output_path='dpo test')

            if not self.use_dpo_weights:
                self.use_dpo_weights = True
        
        # if not self.save_concepts:
        #     self.save_concepts = True

        reward = 50 * alpha * (score1 + score2)

        # if both_positive_and_high:
        #     reward += 10
        # elif both_negative_and_low:
        #     reward -= 10

        self.dpo_epochs += 2

        done = False

        print(f"Action: {current_concept}, Reward: {reward}, Score: {score1}, {score2}, Best Score: {self.best_action_score[action]}")
       # print(f"Action List: {self.action_list}")
        print("====================================================")

        # Add data to a file
        with open("actions.txt", "a") as f:
            f.write(current_concept + "\n")
        
        with open("reward.txt", "a") as f:
            f.write(str(reward) + "\n")
        
        with open("score.txt", "a") as f:
            f.write(str(score1) + " " + str(score2) + "\n")

        if reward <= 0 or len(self.action_list) >= 1:
            self.last_reward = reward
            done = True

        pref_clip_input = clip_processor(text=str(p1) +str(p2), return_tensors="pt", padding=True).to(device)
        obs = clip_model.get_text_features(**pref_clip_input)

        return obs.cpu().detach().numpy(), reward, done, {}


print('Staring RL training')
env = DummyVecEnv([lambda: GenConceptEnv(filtered_keywords)])
dqn_model = DQN("MlpPolicy", env, verbose=1, buffer_size = 1000, exploration_final_eps=0.9, exploration_initial_eps=1.0)

# Update the RL model here
RL_STEP = 500

rl_folder_name = 'RL_checkpoints'
checkpoint_callback = CheckpointCallback(save_freq=100, save_path=rl_folder_name, name_prefix="dqn_model")

dqn_model.learn(total_timesteps=RL_STEP, callback=checkpoint_callback)
dqn_model.save(f"{rl_folder_name}/dqn_model_final.zip")