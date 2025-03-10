#
# This requires the installation of HF transformers, and wandb to work
# Which is not added to the project requirements.txt
#
# ```
# pip3 install transformers datasets wandb
# ```
#
# The trainer is meant to be used with notebooks for testing and validation, hence the the lack of 
# - CLI arguments processing for the trainer
# - LR decay
# - No microbatching
# - No dataset masking
# - Whole dataset is pusehd to memory???
# - No model saving???
# - ????
#

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True

class SimpleTestTrainer:
    def __init__(
            self, model, 

            # The device to use for the training process
            # including the dataset itself
            device="cuda:0",

            # hf_dataset being tested is expected to have a "text" column, within the "train" split
            # there is no validation split, as we will just be using an average loss
            hf_dataset="recursal/SuperWiki-Tiny", 

            # We do not implement proper packing logic, so we will use trim all data records to the
            # indicated context length. Min length is set to -1, which defautls to the ctx length
            # where it filters out all dataset less then the ctx length
            dataset_ctx_length=4096,
            dataset_min_length=-1,

            # Percentage split for the validation dataset
            val_split = 0.001,

            # By default, we use the GPT neox tokenizer, with a vocab size of 50432
            # RWKV models are by default tagged to the world model, which has a vocab size of 65536
            # You will need to initialize the model with the neox vocab size instead
            tokenizer_name="EleutherAI/gpt-neox-20b",

            # Maximum steps to run, and eval between steps
            max_train_steps = 50000,
            val_interval_steps = 1000,
            
            # PS: We are expecting this trainer to only run on 4090's
            batch_size=8, 
            learning_rate=0.001, # 1e-3
            num_epochs=1, 
            project_name="RWKV-Block.SimpleTestTrainer"
        ):

        # Normalize and Log the initial trainer settings
        if dataset_min_length <= -1:
            dataset_min_length = dataset_ctx_length
        
        print("---------------------------------------------")
        print("[SimpleTestTrainer] Initializing the trainer for: ", project_name)
        print("- hf_dataset:         ", hf_dataset)
        print("- dataset_ctx_length: ", dataset_ctx_length)
        print("- dataset_min_length: ", dataset_min_length)
        print("- tokenizer_name:     ", tokenizer_name)
        print("- batch_size:         ", batch_size)
        print("- learning_rate:      ", learning_rate)
        print("- num_epochs:         ", num_epochs)
        print("---------------------------------------------")

        # Save some vars
        self.device = device

        # Load the tokenizer
        print("[SimpleTestTrainer] Loading the tokenizer: ", tokenizer_name, "...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the dataset
        print("[SimpleTestTrainer] Loading the dataset: ", hf_dataset, "...")
        self.dataset = load_dataset(hf_dataset)

        # Prepare the training dataset
        print("[SimpleTestTrainer] Preparing the training dataset...")

        def tokenizer_fn(dataset_batch):
            return self.tokenizer(
                dataset_batch["text"],
                truncation=True,
                max_length=dataset_ctx_length,
                padding="max_length",
                # return_attention_mask=True,
                return_length=True
            )

        dataset_threads = max(int(os.cpu_count()), 1)
        tokenized_dataset = self.dataset['train'].map(
            tokenizer_fn,
            batched=True,
            num_proc=dataset_threads
        ).filter(
            lambda x: x['length'] >= dataset_min_length,
            num_proc=dataset_threads
        ).train_test_split(
            test_size=val_split,
            shuffle=True,
            seed=42 # we fix the seed for reproducibility
        )

        self.train_dataset = tokenized_dataset['train']
        self.val_dataset = tokenized_dataset['test']

        # Log the test sizes
        print("[SimpleTestTrainer] Training dataset size:   ", len(self.train_dataset))
        print("[SimpleTestTrainer] Validation dataset size: ", len(self.val_dataset))

        # Prepare the data loaders
        print("[SimpleTestTrainer] Preparing the data loaders...")

        def collate_fn(batch):
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            return { 'input_ids': input_ids}
            # attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            # return { 'input_ids': input_ids, 'attention_mask': attention_mask }

        # Trim down max steps
        self.val_interval_steps = val_interval_steps
        if max_train_steps > - 1:
            self.train_loader = DataLoader(
                # Get the max_train_steps worth of dataset 
                self.train_dataset.select(range(max_train_steps * batch_size)),
                batch_size=batch_size,
                collate_fn=collate_fn,
                pin_memory=True,
                pin_memory_device=device,
                drop_last=True
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                pin_memory=True,
                pin_memory_device=device,
                drop_last=True
            )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            pin_memory_device=device,
            drop_last=True
        )

        # Log the batch sizes
        print("[SimpleTestTrainer] Training batch count:   ", len(self.train_loader))
        print("[SimpleTestTrainer] Validation batch count: ", len(self.val_loader))

        # Setting up optimizer and loss function
        print("[SimpleTestTrainer] Setting up the optimizer, loss function...")
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

        # Initialize wandb
        print("[SimpleTestTrainer] Initializing wandb...")

        # Check if wandb is logged in
        # if wandb.run is not None:
        if wandb.login():
            print("[SimpleTestTrainer] wandb is logged in.")
            wandb.init(project=project_name, config={
                "model_class": model.__class__.__name__,
                "dataset": hf_dataset,
                "tokenizer": tokenizer_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "dataset_ctx_length": dataset_ctx_length,
                "dataset_min_length": dataset_min_length
            })
        else:
            print("[SimpleTestTrainer] skipping wandb - not initialized.")

        print("[SimpleTestTrainer] Initialization complete.")
        print("---------------------------------------------")

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        steps = 0
        progress = tqdm(self.train_loader, desc="Training")
        for batch in progress:
            input_ids = batch['input_ids'].to(self.device)
            label = input_ids[:,1:].reshape(-1).clone()

            self.optimizer.zero_grad()

            # If using tmix_backend with cuda
            if "cuda" in self.model.configMap.tmix_backend:
                # Cuda has some issues with compile now sadly
                batch_t_logits, fwd_state = self.model.forward(input_ids)
            else:
                # Run with compiler
                ini_state = self.model.get_init_state(input_ids.shape[0])
                batch_t_logits, fwd_state = self.model.forward_with_reduce_compile(input_ids, ini_state)

            assert torch.isnan(input_ids).sum() == 0,      "NaN detected in the input"
            assert torch.isnan(label).sum() == 0,          "NaN detected in the label"
            assert torch.isnan(batch_t_logits).sum() == 0, "NaN detected in the model output"

            index = batch_t_logits[:,:-1,:].reshape(-1, batch_t_logits.size(-1))
            loss = self.criterion(index, label)

            assert torch.isnan(loss).sum() == 0, "Loss is NaN"
            
            loss.backward()
            self.optimizer.step()

            # Logging
            progress.set_postfix(loss=loss.item())

            # val_interval_steps defaults to 10000
            val_interval_steps = self.val_interval_steps

            # Call the validation step every 10k steps
            if val_interval_steps > 0 and steps % val_interval_steps == 0:
                val_loss = self.run_validation(progress)
                progress.set_postfix(last_val_loss=val_loss)

                if wandb.run is not None:
                    wandb.log({"train_loss": loss.item(), "val_loss": val_loss})
            else:
                if wandb.run is not None:
                    wandb.log({"train_loss": loss.item()})

            steps += 1
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def run_validation(self, prg):
        if not self.val_loader:
            return None

        self.model.eval()
        total_loss = 0

        # progress = tqdm(self.val_loader, desc="Validating")
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                label = input_ids[:,1:].reshape(-1).clone()

                # If using tmix_backend with cuda
                if "cuda" in self.model.configMap.tmix_backend:
                    # Cuda has some issues with compile now sadly
                    batch_t_logits, fwd_state = self.model.forward(input_ids)
                else:
                    # Run with compiler
                    ini_state = self.model.get_init_state(input_ids.shape[0])
                    batch_t_logits, fwd_state = self.model.forward_with_reduce_compile(input_ids, ini_state)

                assert torch.isnan(input_ids).sum() == 0,      "NaN detected in the input"
                assert torch.isnan(label).sum() == 0,          "NaN detected in the label"
                assert torch.isnan(batch_t_logits).sum() == 0, "NaN detected in the model output"

                index = batch_t_logits[:,:-1,:].reshape(-1, batch_t_logits.size(-1))
                loss = self.criterion(index, label)

                assert torch.isnan(loss).sum() == 0, "Loss is NaN"
                
                if prg is not None:
                    prg.set_postfix(step_val_loss=loss.item())

                # # Log to wandb
                # if wandb.run is not None:
                #     wandb.log({"val_loss": loss.item()})
                
                total_loss += loss.item()

        mean_val_loss = total_loss / len(self.val_loader)

        # # Log the validation loss
        # print(f"Validation Loss: {mean_val_loss:.4f}")
        # if wandb.run is not None:
        #     wandb.log({"val_loss": loss.item()})

        return mean_val_loss

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            epoch_train_loss = self.train_epoch()
            epoch_val_loss = self.run_validation()

            print(f"Training Loss: {epoch_train_loss:.4f}")
            log_dict = {"epoch_train_loss": epoch_train_loss, "epoch": epoch + 1, "epoch_val_loss": epoch_val_loss}

            # Log to wandb
            if wandb.run is not None:
                wandb.log(log_dict)
            print()

        if wandb.run is not None:
            wandb.finish()

    # def load_best_model(self):
    #     self.model.load_state_dict(torch.load("best_model.pth"))
    #     self.model.to(self.device)
    #     print("Loaded best model")

# Usage example:
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
# trainer = SimpleTrainer(model, dataset_name="wikitext", tokenizer_name="EleutherAI/gpt-neox-20b", project_name="my_language_model")
# trainer.train()
# trainer.load_best_model()