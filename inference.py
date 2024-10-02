import config as sc
import torch
import gc, random, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
torch.backends.cuda.matmul.allow_tf32 = True

# Keywords Imports
import os
from PIL import Image
from collections import Counter, defaultdict
from transformers import BlipProcessor, BlipForQuestionAnswering, CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
import spacy

# TCAV Imports
import torchvision
from torchvision import transforms
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator

from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str
from captum.concept._utils.classifier import Classifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Diffusion Imports
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import CLIPTextModel
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datasets import Dataset
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import math
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import (convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available

# RL Imports
import gym
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


""" TCAV Score """
class CustomImageDataset(IterableDataset):
    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform
    
    def __iter__(self) -> Iterator[Tensor]:
        for img in self.img_list:
            yield self.transform(img)

def create_random_concept(num_images=10):
    if not os.path.exists(sc.RANDOM_FOLDER_NAME):
        folders = os.listdir(sc.TRAIN_FOLDER)
        folders.remove(str(sc.CLASS_NAME))
        random_folders = random.sample(folders, num_images)
        random_imgs = []

        for folder in random_folders:
            random_img = random.sample(os.listdir(os.path.join(sc.TRAIN_FOLDER, folder)), 1)[0]
            img_path = os.path.join(sc.TRAIN_FOLDER, folder, random_img)
            random_imgs.append(img_path)
        
        os.makedirs(sc.RANDOM_FOLDER_NAME, exist_ok=True)
        for img in random_imgs:
            shutil.copy(img, sc.RANDOM_FOLDER_NAME)
    
    random_imgs_check = os.listdir(sc.RANDOM_FOLDER_NAME)
    if len(random_imgs_check) != num_images:
        shutil.rmtree(sc.RANDOM_FOLDER_NAME)
        create_random_concept(num_images=num_images)

def transform_image(img):
    return sc.TRANSFORMS(img)

def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return sc.TRANSFORMS(img)

def load_image_tensors(class_name, root_path=sc.TRAIN_FOLDER, is_transform=True):
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.JPEG')

    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        tensors.append(sc.TRANSFORMS(img) if is_transform else img)
    
    return tensors

def load_image_PIL(class_name, root_path=sc.TRAIN_FOLDER):
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.JPEG')

    images = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        images.append(img)
    
    return images

def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform_image(img)

def assemble_concept_filename(name, id, concept_path=None):
    if concept_path is None:
        concept_path = name + '/'
    else:
        concept_path = os.path.join(concept_path, name) + '/'
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)

def assemble_concept_image(name, id, concept_imgs):
    dataset = CustomImageDataset(concept_imgs, sc.TRANSFORMS)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)

class CustomClassifier(Classifier):
    def __init__(self):
        self.lm = linear_model.LogisticRegression(max_iter=1000)
        self.test_size = 0.33
        self.evaluate_test = False
        self.metrics = None

    def train_and_eval(self, dataloader: DataLoader, **kwargs):
        inputs, labels = [], []
        for X, y in dataloader:
            inputs.append(X)
            labels.append(y)
        inputs, labels = torch.cat(inputs).detach().cpu().numpy(), torch.cat(labels).detach().cpu().numpy()
        if self.evaluate_test:
            X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=self.test_size)
        else:
            X_train, y_train = inputs, labels
        self.lm.fit(X_train, y_train)

        if self.evaluate_test:
            self.metrics = {'accs': self.lm.score(X_test, y_test)}
            return self.metrics
        self.metrics = {'accs': self.lm.score(X_train, y_train)}
        return self.metrics

    def weights(self):
        if len(self.lm.coef_) == 1:
            # if there are two concepts, there is only one label.
            # We split it in two.
            return torch.tensor(np.array([-1 * self.lm.coef_[0], self.lm.coef_[0]]))
        else:
            return torch.tensor(self.lm.coef_)
    
    def classes(self):
        return self.lm.classes_

    def get_metrics(self):
        return self.metrics
    
def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def get_reward(scores, layers, concepts):
    keys = [key for key in scores.keys()]

    reward = scores[keys[0]][layers[0]]['sign_count'][0] 
    return reward

def remove_cav_folder():
    if os.path.exists('cav'):
        shutil.rmtree('cav')


actions = []
with open(f'{sc.DPO_WEIGHTS_PATH}/keywords.txt', 'r') as f:
    for line in f:
        actions.append(line.strip())

""" Diffusion Functions and DPO Code """
class GenConceptGenerator():
    def __init__(self, model_path, output_dir, seed=None):
        self.model_name = model_path
        self.output_path = output_dir
        self.seed = seed
        self.beta_dpo = 2500 # DPO KL divergence penalty

        # Checking the output directory
        os.makedirs(self.output_path, exist_ok=True)
        # Checking the checkpoints directory
        os.makedirs(self.output_path + "/checkpoints", exist_ok=True)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=sc.GRADIENT_STEP_ACCUMULATION,
            mixed_precision="fp16",
            project_config=ProjectConfiguration(project_dir=self.output_path),
        )

    def generate_concept(self, prompts, with_lora_dpo=False):
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Generate concept
        pipeline = DiffusionPipeline.from_pretrained(self.model_name, torch_dtype=weight_dtype)
        if with_lora_dpo:
            print("Using LoRA DPO")
            pipeline.load_lora_weights(self.output_path, weight_name="lora_dpo_weights.safetensors")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        # pipeline = pipeline.to(self.accelerator.device)
        pipeline = pipeline.to(self.accelerator.device, dtype=weight_dtype)
        pipeline.enable_vae_slicing()
        # pipeline.enable_sequential_cpu_offload()
        if is_xformers_available():
           pipeline.enable_xformers_memory_efficient_attention()
        
        pipeline.safety_checker = None

        # with torch.cuda.amp.autocast():
        image = pipeline(prompts, num_inference_steps=sc.NUM_INFERENCE_STEPS, height=sc.GEN_IMAGE_SIZE, width=sc.GEN_IMAGE_SIZE).images
        
        # Free up CUDA memory
        self.accelerator.free_memory()
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
        return image

    def tokenize_text(self, tokenizer, prompts):
        max_length = tokenizer.model_max_length
        text_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        return text_inputs.input_ids
    
    @torch.no_grad()
    def encode_prompt(self, text_encoder, input_ids):
        text_encoder_ids = input_ids.to(text_encoder.device)
        attention_mask = None
        prompt_embeds = text_encoder(text_encoder_ids, attention_mask=attention_mask)[0]
        return prompt_embeds
    
    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(self.unwrap_model(self.unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError("Unexpected save model: ", {model.__class__})
                
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
            
            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=None,
            )
    
    def inject_lora_into_unet(self, state_dict, unet, network_alphas=None, adapter_name="default"):
        keys = list(state_dict.keys())
        unet_keys = [k for k in keys if k.startswith("unet.")]
        state_dict = {k.replace("unet.", ""): v for k, v in state_dict.items() if k in unet_keys}
        state_dict = convert_unet_state_dict_to_peft(state_dict)

        if network_alphas is not None:
            alpha_keys = [k for k in network_alphas.keys() if k.startswith("unet")]
            network_alphas = {k.replace("unet.", ""): v for k, v in network_alphas.items() if k in alpha_keys}
            network_alphas = convert_unet_state_dict_to_peft(network_alphas)
        
        set_peft_model_state_dict(unet, state_dict, adapter_name)
        unet.load_attn_procs(state_dict, network_alphas=network_alphas)

    def load_model_hook(self, models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(self.unwrap_model(self.unet))):
                unet_ = model
            else:
                raise ValueError("Unexpected load model: ", {model.__class__})
        
        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        self.inject_lora_into_unet(lora_state_dict, unet_, network_alphas=network_alphas)
        
        models = [unet_]
        for model in models:
            for p in model.parameters():
                # Upcast trainable parameters (LoRA) into fp32
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)

    def update_pipeline(self, dpo_preferred_concepts, dpo_unpreferred_concepts, prompts, num_epochs=3, continue_training=False, output_file_name="lora_dpo_weights.safetensors"):
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, subfolder="tokenizer", use_fast=False)

        # Getting text encoder class
        text_encoder_config = PretrainedConfig.from_pretrained(self.model_name, subfolder="text_encoder")
        text_encoder_class = CLIPTextModel if text_encoder_config.architectures[0] == "CLIPTextModel" else ValueError("Model not supported")

        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_name, subfolder="scheduler")
        text_encoder = text_encoder_class.from_pretrained(self.model_name, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.model_name, subfolder="unet")

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        vae.to(self.accelerator.device, dtype=weight_dtype)
        text_encoder.to(self.accelerator.device, dtype=weight_dtype)

        # Setup LoRA
        unet_lora_config = LoraConfig(
            r = 8,
            lora_alpha=8,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        )
        self.unet.add_adapter(unet_lora_config)
        # Upcast trainable parameters (LoRA) into fp32
        for p in self.unet.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)
        
        # Multi-GPU training mode
        self.unet.enable_gradient_checkpointing()
    
        # # Efficient training
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        
        self.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.load_model_hook)
        
        # Optimizer creation
        optimizer_class = torch.optim.AdamW
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        # Values from the huggingface github page
        optimizer = optimizer_class(
            params_to_optimize,
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08
        )

        # Dataset and dataloader creation
        data_dict = {
            "prompt": prompts,
            "preferred": dpo_preferred_concepts, # jpg_0
            "unpreferred": dpo_unpreferred_concepts, # jpg_1
            "label_0": [1] * len(dpo_preferred_concepts),
            "label_1": [0] * len(dpo_unpreferred_concepts)
        }

        train_dataset = Dataset.from_dict(data_dict)
        train_transform = transforms.Compose([
            transforms.Resize(int(512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        def preprocess_train_data(data):
            all_pixel_values = []

            for key in ["preferred", "unpreferred"]:
                images = [img if isinstance(img, Image.Image) else Image.open(img).convert("RGB") for img in data[key]]
                pixel_values = [train_transform(image) for image in images]
                all_pixel_values.append(pixel_values)
            
            # Double the channel dimentions, preferred then unpreferred
            im_tup_iterator = zip(*all_pixel_values)
            combined_pixel_values = []

            for im_tup, label_0 in zip(im_tup_iterator, data["label_0"]):
                if label_0 == 0:
                    im_tup = im_tup[::-1]
                combined_im = torch.cat(im_tup, dim=0)
                combined_pixel_values.append(combined_im)
            data["pixel_values"] = combined_pixel_values
            data["input_ids"] = self.tokenize_text(tokenizer, data["prompt"])
            return data
        
        with self.accelerator.main_process_first():
            train_dataset = train_dataset.shuffle(seed=self.seed)
            train_dataset = train_dataset.with_transform(preprocess_train_data)
        
        def collate_fn(data):
            pixel_values = torch.stack([x["pixel_values"] for x in data])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            final_dict = {"pixel_values": pixel_values}
            final_dict["input_ids"] = torch.stack([x["input_ids"] for x in data])
            return final_dict
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1, # Might need to change this
            shuffle=True,
            collate_fn=collate_fn
        )

        # Training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / sc.GRADIENT_STEP_ACCUMULATION)
        num_epochs = num_epochs
        max_train_steps = num_epochs * num_update_steps_per_epoch

        # Scheduler
        lr_scheduler = get_scheduler(
            "constant", # Type of scheduler
            optimizer=optimizer,
            num_warmup_steps= 0 * self.accelerator.num_processes,
            num_training_steps = max_train_steps * self.accelerator.num_processes,
            num_cycles=1,
            power=1.0
        )
        self.unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.unet, optimizer, train_dataloader, lr_scheduler
        )

        # Recalculate the training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / sc.GRADIENT_STEP_ACCUMULATION) # 25 is gradient accumulation steps
        num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # Train!
        global_step = 0
        first_epoch = 0

        # Loading weights for continued training
        if continue_training:
            dirs = os.listdir(self.output_path + "/checkpoints")
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
            path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print("No checkpoint found, training from scratch")
                initial_global_step = 0
            else:
                self.accelerator.print(f"Loading checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.output_path + "/checkpoints", path))
                global_step = int(path.split("-")[-1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch

                # Delete the other checkpoints except the last one
                for d in dirs[:-1]:
                    shutil.rmtree(os.path.join(self.output_path + "/checkpoints", d))
                
        else:
            initial_global_step = 0
        
        progress_bar = tqdm(
            range(0, max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process
        )

        self.unet.train()
        for e in range(first_epoch, num_epochs):
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                    feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1))

                    latents = []
                    for i in range(0, feed_pixel_values.shape[0], 2): # 2 is VAE batch size
                        latents.append(
                            vae.encode(feed_pixel_values[i:i+2]).latent_dist.sample()
                        )
                    latents = torch.cat(latents, dim=0)
                    latents = latents * vae.config.scaling_factor

                    # Sample noise to be added to the latents
                    noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)

                    # Sample a random timestep for each image
                    bsz = latents.shape[0] // 2
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long
                    ).repeat(2)

                    # Add noise to the model input according to the scheduler
                    # (Forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get text embeddings
                    encoder_hidden_states = self.encode_prompt(text_encoder, batch["input_ids"]).repeat(2, 1, 1)

                    # Predict the noise residuals
                    model_pred = self.unet(
                        noisy_model_input, 
                        timesteps, 
                        encoder_hidden_states
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # Compute losses
                    model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                    model_diff = model_losses_w - model_losses_l

                    # Reference model prediction
                    self.accelerator.unwrap_model(self.unet).disable_adapters()
                    with torch.no_grad():
                        ref_preds = self.unet(
                            noisy_model_input, 
                            timesteps, 
                            encoder_hidden_states
                        ).sample.detach()
                        ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
                        ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

                        ref_loss_w, ref_loss_l = ref_loss.chunk(2)
                        ref_diff = ref_loss_w - ref_loss_l
                        raw_ref_loss = ref_loss.mean()
                    
                    # Re-enable adapters
                    self.accelerator.unwrap_model(self.unet).enable_adapters()

                    # Final loss
                    logits = ref_diff - model_diff
                    loss = -1 * F.logsigmoid(self.beta_dpo * logits).mean() # Sigmoid loss

                    implicit_acc = (logits > 0).sum().float() / logits.size(0)
                    implicit_acc += 0.5 * (logits == 0).sum().float() / logits.size(0)

                    # Backward pass
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if self.accelerator.is_main_process:
                        save_path = os.path.join(self.output_path, f"checkpoints/checkpoint-{global_step}")
                        self.accelerator.save_state(output_dir=save_path)

                logs = {
                    "loss": loss.detach().item(),
                    "raw_model_loss": raw_model_loss.detach().item(),
                    "ref_loss": raw_ref_loss.detach().item(),
                    "implicit_acc": implicit_acc.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= max_train_steps:
                    break

        # Save the LoRA layers
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.unet = self.accelerator.unwrap_model(self.unet)
            self.unet = self.unet.to(torch.float32)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet))

            LoraLoaderMixin.save_lora_weights(
                save_directory=self.output_path,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=None,
                weight_name=output_file_name
            )
        
        self.accelerator.end_training()
        
        # Free up CUDA memory
        self.accelerator.free_memory()
        del train_dataloader, optimizer, lr_scheduler, self.unet, vae, text_encoder, noise_scheduler
        torch.cuda.empty_cache()
        gc.collect()


""" RL Environment and Training """

actions = actions
total_class_images = len(os.listdir(os.path.join(sc.TRAIN_FOLDER, sc.CLASS_NAME)))
num_class_images = sc.N_SAMPLES

action_lifetime = []
reward_lifetime = []

create_random_concept(num_images=sc.N_SAMPLES)
classifier = CustomClassifier()

class_imgs = load_image_tensors(sc.CLASS_NAME)
class_tensor = torch.stack(class_imgs)

class GenConceptEnv(gym.Env):
    def __init__(self):
        super(GenConceptEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(len(actions))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(num_class_images, 224, 224, 3), dtype=np.float32)

        self.action_list = []
        self.save_concepts = False
        self.last_reward = 0
        self.use_dpo_weights = False
        self.dpo_epochs = 2
        self.idx = 1

        self.n_class_tensor = None

        self.best_action_score = [-1.0] * len(actions)

    def reset(self):
        if self.save_concepts: # and self.last_reward >= 0
            sample_prompt = f"{', '.join(self.action_list)}"
            generator = GenConceptGenerator(
                model_path=sc.GEN_MODEL_NAME,
                output_dir=sc.DPO_WEIGHTS_PATH
            )
            concepts = generator.generate_concept([sample_prompt] * 10, with_lora_dpo=self.use_dpo_weights)
            sample_folder = f"{' '.join(self.action_list)}_samples {self.idx}"
            sample_path = os.path.join("test_concepts", sample_folder)
            os.makedirs(sample_path, exist_ok=True)

            for i, concept in enumerate(concepts):
                concept.save(os.path.join(sample_path, f"{i+1}.jpg"))
            
            torch.cuda.empty_cache()
            gc.collect()
        
        # self.idx = 1
        self.action_list = []
        
        # Delete cav folder
        remove_cav_folder()

        all_class_imgs = class_imgs
        n_class_imgs = random.sample(all_class_imgs, num_class_images)
        self.n_class_tensor = torch.stack(n_class_imgs)

        obs = self.n_class_tensor.numpy()
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        obs = obs.transpose(0, 2, 3, 1)
        return obs

    def step(self, action):
        current_concept = actions[action]
        self.action_list.append(current_concept)

        count_action = self.action_list.count(current_concept)
        alpha = 2*count_action/sc.TOTAL_RL_STEP

        prompt = f"{', '.join(self.action_list)}."

        generator = GenConceptGenerator(
            model_path=sc.GEN_MODEL_NAME,
            output_dir=sc.DPO_WEIGHTS_PATH
        )
        images = generator.generate_concept([prompt] * sc.N_SAMPLES * 2, with_lora_dpo=self.use_dpo_weights)
        random.shuffle(images)

        c1 = images[:sc.N_SAMPLES]
        c2 = images[sc.N_SAMPLES:]

        tcav_obj = TCAV(
            model=sc.MODEL_TO_EXPLAIN,
            layers=sc.LAYERS,
            classifier=classifier,
            layer_attr_method = LayerIntegratedGradients(
                sc.MODEL_TO_EXPLAIN,
                None,
                multiply_by_inputs=False
            )
        )

        random_concepts = assemble_concept_filename(sc.RANDOM_FOLDER_NAME, 0)

        gen_c1 = assemble_concept_image(name=prompt, id=1, concept_imgs=c1)
        gen_c2 = assemble_concept_image(name=prompt, id=2, concept_imgs=c2)

        exp_set_1 = [[gen_c1, random_concepts]]
        exp_set_2 = [[gen_c2, random_concepts]]

        tcav_score_1 = tcav_obj.interpret(
            inputs=self.n_class_tensor,
            experimental_sets=exp_set_1,
            target=sc.CLASS_TO_BE_EXPLAINED,
            n_steps=5,
        )

        tcav_score_2 = tcav_obj.interpret(
            inputs=self.n_class_tensor,
            experimental_sets=exp_set_2,
            target=sc.CLASS_TO_BE_EXPLAINED,
            n_steps=5,
        )

        score1 = get_reward(tcav_score_1, sc.LAYERS, exp_set_1)
        score2 = get_reward(tcav_score_2, sc.LAYERS, exp_set_2)

        both_positive_and_high = False

        if score1 > 0.5 and score2 > 0.5:
            alpha = 1
            both_positive_and_high = True

        if score1 > score2:
            best_score = score1
            preferred_concepts = c1
            unpreferred_concepts = c2
            if score1 >= self.best_action_score[action]:
                self.best_action_score[action] = score1
        else:
            best_score = score2
            preferred_concepts = c2
            unpreferred_concepts = c1
            if score2 >= self.best_action_score[action]:
                self.best_action_score[action] = score2
        
        # No DPO Needed in Inference
        # if (best_score == self.best_action_score[action]) and not both_positive_and_high:
        #     generator.update_pipeline(preferred_concepts, unpreferred_concepts, [prompt] * sc.N_SAMPLES, num_epochs=self.dpo_epochs, continue_training=self.use_dpo_weights)

        #     if not self.use_dpo_weights:
        #         self.use_dpo_weights = True

        #     self.dpo_epochs += 2
        
        if not self.save_concepts:
            self.save_concepts = True

        reward = 50 * alpha * (score1 + score2)

        self.idx += 1

        done = False

        print(f"Action: {current_concept}, Reward: {reward}, Score: {score1}, {score2}, Best Score: {self.best_action_score[action]}")
        print(f"Action List: {self.action_list}")
        print("====================================================")

        if reward <= 0 or len(self.action_list) >= 1:
            self.last_reward = reward
            done = True
        
        # Free up memory
        del generator
        torch.cuda.empty_cache()
        gc.collect()
        
        obs = []
        for img in preferred_concepts:
            obs.append(sc.TRANSFORMS(img))
        obs = torch.stack(obs).numpy()

        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        obs = obs.transpose(0, 2, 3, 1)

        return obs, reward, done, {}


env = DummyVecEnv([lambda: GenConceptEnv()])
dqn_model = DQN.load(f"{sc.DPO_WEIGHTS_PATH}/saved_rl_models/dqn_models_./dqn_model_final.zip", env=env)

action_list = []
for e in range(10000):
    obs = env.reset()
    env.seed(0)
    done = False
    reward = 0

    # while not done:
    action, _states = dqn_model.predict(obs)
    action_list.append(actions[action[0]])

    if (e+1) % 1000 == 0:
        print(f"Episode: {e+1}")


from collections import Counter

# Count the frequency of each string
frequency_counter = Counter(action_list)

# Sort the strings by their frequency in ascending order
sorted_strings = sorted(action_list, key=lambda x: (frequency_counter[x], x))

# Get the unique strings
unique_sorted_strings = list(dict.fromkeys(sorted_strings))

# Print the unique list in ascending order of frequency
print("===============================================================")
print('Top 5 Actions:')
print(unique_sorted_strings[:5])


# Create a folder to save all the plots
os.makedirs(f"{sc.DPO_WEIGHTS_PATH}/plots", exist_ok=True)

# Save the concept order in txt file
with open(f"{sc.DPO_WEIGHTS_PATH}/plots/concept_order.txt", "w") as f:
    for item in unique_sorted_strings:
        f.write("%s\n" % item)


all_generated_images = {}
all_scores = {}

for i in unique_sorted_strings[:5]:
    print(f"Generating images for {i}")

    for j in range(5):
        print(f"Generating image {j+1}")

        gen = GenConceptGenerator(
            model_path=sc.GEN_MODEL_NAME,
            output_dir=sc.DPO_WEIGHTS_PATH
        )
        images = gen.generate_concept([i] * 10, with_lora_dpo=True)

        del gen
        torch.cuda.empty_cache()
        gc.collect()

        all_class_imgs = class_imgs
        n_class_imgs = random.sample(all_class_imgs, num_class_images)
        n_class_tensor = torch.stack(n_class_imgs)

        remove_cav_folder()

        tcav_obj = TCAV(
                    model=sc.MODEL_TO_EXPLAIN,
                    layers=sc.LAYERS,
                    classifier=classifier,
                    layer_attr_method = LayerIntegratedGradients(
                        sc.MODEL_TO_EXPLAIN,
                        None,
                        multiply_by_inputs=False
                    )
                )

        random_concepts = assemble_concept_filename(sc.RANDOM_FOLDER_NAME, 0)

        gen_c1 = assemble_concept_image(name="prompt", id=1, concept_imgs=images)

        exp_set_1 = [[gen_c1, random_concepts]]

        tcav_score_1 = tcav_obj.interpret(
            inputs=n_class_tensor,
            experimental_sets=exp_set_1,
            target=sc.CLASS_TO_BE_EXPLAINED,
            n_steps=5,
        )

        score1 = get_reward(tcav_score_1, sc.LAYERS, exp_set_1)

        all_generated_images[f"{i}_{j}"] = images
        all_scores[f"{i}_{j}"] = score1



def get_tcav_range_plot(tcav_scores, labels, colors, name=None):
    base_path = f"{sc.DPO_WEIGHTS_PATH}/plots"
    os.makedirs(base_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_ylim(0, 1)
    ax.set_xlim(-1, len(labels))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    for i, label in enumerate(labels):
        mean = np.mean(tcav_scores[i])
        std_dev = np.std(tcav_scores[i])

        ax.vlines(x=i, ymin=0, ymax=1, colors='gray', linestyles='dashed', lw=1)

        rect_width = std_dev * 2
        rect = patches.Rectangle((i - 0.1, mean - rect_width / 2), 0.2, rect_width, color=colors[i], alpha=1, label=label)
        ax.add_patch(rect)

        ax.text(i, mean + rect_width/2 + 0.02, f'{mean:.3f}', horizontalalignment='center', verticalalignment='center', color=colors[i], fontsize=10)
    
    ax.set_ylabel('TCAV Score')
    ax.set_xlabel('Concepts')
    ax.legend(title="Concepts")

    plt.tight_layout()

    if name is not None:
        plt.savefig(f"{base_path}/{name}.pdf", format='pdf')
    
    plt.show()
grouped_data = defaultdict(list)

# Parse and group the data by the prefix (everything before '_')
for key, value in all_scores.items():
    prefix = key.rsplit('_', 1)[0]  # Split by underscore and get the prefix part
    grouped_data[prefix].append(float(value))  # Append the value to the corresponding prefix list

# Convert the defaultdict to a 2D list (each sublist has a prefix and its grouped values)
tcav_result = [values for prefix, values in grouped_data.items()]
tcav_prefixes = [prefix for prefix, values in grouped_data.items()]

get_tcav_range_plot(tcav_result, tcav_prefixes, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], name="tcav_range_plot")

# Function to save generated concept images
def save_concept_images(images, concept_name):
    base_path = f"{sc.DPO_WEIGHTS_PATH}/plots/generated_images/{concept_name}"
    os.makedirs(base_path, exist_ok=True)

    for i, img in enumerate(images):
        img.save(f"{base_path}/{i}.jpg")

class_pil_imgs = load_image_PIL(sc.CLASS_NAME)
random_class_pil_imgs = class_pil_imgs


# Combine similar prefix images into single tag
grouped_images = defaultdict(list)

for key, value in all_generated_images.items():
    prefix = key.rsplit('_', 1)[0]
    grouped_images[prefix].extend(value)

# Save the generated images
for key, value in grouped_images.items():
    save_concept_images(value, key)