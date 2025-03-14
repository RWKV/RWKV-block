import math
import re
import string
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Any, Union
from threading import Lock
import torch.multiprocessing as mp

# set TOKENIZERS_PARALLELISM to False
# as it resolves most tokenizer race condition issues.
#
# This is critical for getting reliable multi-GPU training right
# due to the amount of wierd errors with various tokenizers
#
# we do a single thread safe tokenization lock anyway
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TemplateFormatter:
    """
    A custom template formatter that extends Python's string formatting capabilities.
    
    Features:
    1. Support for nested dictionary access using dot notation (e.g., {metadata.author})
    2. Support for converting numeric values to alphabetic characters (e.g., {to_letter(score)})
    """
    
    # Regular expression to find placeholders in templates
    # Matches both standard placeholders {key} and function calls {func_name(arg1, arg2, ...)}
    PLACEHOLDER_PATTERN = re.compile(r'\{([^{}]+)\}')
    
    # Regular expression to match function calls within placeholders
    FUNCTION_PATTERN = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)')
    
    @staticmethod
    def get_nested_value(data: Dict[str, Any], key_path: str) -> Any:
        """
        Access nested dictionary values using dot notation.
        
        Args:
            data: The dictionary to access
            key_path: The path to the value, using dot notation (e.g., "metadata.author")
            
        Returns:
            The value at the specified path
            
        Raises:
            KeyError: If any part of the path doesn't exist
        """
        # Handle array indexing with square brackets
        if '[' in key_path and ']' in key_path:
            # Split at the first opening bracket
            base_key, rest = key_path.split('[', 1)
            
            # Get the base value
            if '.' in base_key:
                # If the base key contains dots, it's a nested path
                current = TemplateFormatter.get_nested_value(data, base_key)
            else:
                # Otherwise, it's a direct key
                current = data[base_key]
            
            # Extract the index and any remaining path
            index_str, rest = rest.split(']', 1)
            index = int(index_str)
            
            # Get the value at the index
            current = current[index]
            
            # If there's more to the path, continue recursively
            if rest.startswith('.') and len(rest) > 1:
                return TemplateFormatter.get_nested_value(current, rest[1:])
            return current
        
        # Handle regular dot notation
        if '.' in key_path:
            parts = key_path.split('.', 1)
            return TemplateFormatter.get_nested_value(data[parts[0]], parts[1])
        
        # Base case: direct key access
        return data[key_path]
    
    @staticmethod
    def to_letter(number: Union[int, float], case: str = 'upper') -> str:
        """
        Convert a number to an alphabetic character.
        
        Args:
            number: The number to convert (0-25 maps to A-Z or a-z)
            case: 'upper' for uppercase, 'lower' for lowercase
            
        Returns:
            The corresponding letter
        """
        try:
            # Convert to integer
            num = int(number)
            
            # Handle numbers outside the range
            if num < 0:
                return ''
            
            # For numbers > 25, use multiple letters (A-Z, then AA, AB, etc.)
            result = ''
            while True:
                remainder = num % 26
                result = string.ascii_uppercase[remainder] + result
                num = num // 26 - 1  # Subtract 1 to make it 0-based
                if num < 0:
                    break
            
            # Apply case
            if case.lower() == 'lower':
                return result.lower()
            return result
        except (ValueError, TypeError):
            # Return empty string for invalid inputs
            return ''
    
    @classmethod
    def format_template(cls, template: str, data: Dict[str, Any]) -> str:
        """
        Format a template string using the provided data.
        
        Args:
            template: The template string with placeholders
            data: The data dictionary to use for formatting
            
        Returns:
            The formatted string
        """
        def replace_placeholder(match):
            placeholder = match.group(1)
            
            # Check if it's a function call
            func_match = cls.FUNCTION_PATTERN.match(placeholder)
            if func_match:
                func_name = func_match.group(1)
                args_str = func_match.group(2)
                
                # Parse arguments
                args = []
                kwargs = {}
                
                if args_str.strip():
                    # Split by commas, but respect nested structures
                    parts = []
                    current = ''
                    nesting_level = 0
                    
                    for char in args_str:
                        if char == ',' and nesting_level == 0:
                            parts.append(current.strip())
                            current = ''
                        else:
                            if char in '({[':
                                nesting_level += 1
                            elif char in ')}]':
                                nesting_level -= 1
                            current += char
                    
                    if current:
                        parts.append(current.strip())
                    
                    # Process each argument
                    for part in parts:
                        if '=' in part:
                            # It's a keyword argument
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Try to evaluate the value
                            try:
                                # First check if it's a string literal
                                if (value.startswith("'") and value.endswith("'")) or \
                                   (value.startswith('"') and value.endswith('"')):
                                    value = value[1:-1]
                                # Otherwise, try to get it from the data
                                else:
                                    value = cls.get_nested_value(data, value)
                            except:
                                # If evaluation fails, use the literal value
                                pass
                            
                            kwargs[key] = value
                        else:
                            # It's a positional argument
                            # Try to evaluate it
                            try:
                                # First check if it's a string literal
                                if (part.startswith("'") and part.endswith("'")) or \
                                   (part.startswith('"') and part.endswith('"')):
                                    args.append(part[1:-1])
                                # Otherwise, try to get it from the data
                                else:
                                    args.append(cls.get_nested_value(data, part))
                            except:
                                # If evaluation fails, use the literal value
                                args.append(part)
                
                # Call the appropriate function
                if func_name == 'to_letter':
                    if not args and not kwargs:
                        return ''
                    
                    number = args[0] if args else kwargs.get('number', 0)
                    case = kwargs.get('case', 'upper')
                    return cls.to_letter(number, case)
                
                # Unknown function
                return f"[Unknown function: {func_name}]"
            
            # It's a regular placeholder, try to get the value using nested access
            try:
                return str(cls.get_nested_value(data, placeholder))
            except (KeyError, TypeError, IndexError):
                # Return an empty placeholder for missing keys
                return f"[Missing: {placeholder}]"
        
        # Replace all placeholders in the template
        return cls.PLACEHOLDER_PATTERN.sub(replace_placeholder, template)

class TextDatasetStreamer(IterableDataset):
    """
    A PyTorch IterableDataset for streaming and processing HuggingFace Text datasets.

    This class handles loading, preprocessing, and batching of multiple HuggingFace datasets
    for use in training language models. It supports dataset interleaving, sharding for
    distributed training, and dynamic batch packing.

    Templates:
    The class uses a template system to format dataset rows before tokenization. Templates
    are specified in the dataset_configs parameter and use an enhanced version of Python's 
    string formatting syntax with named placeholders corresponding to dataset column names.
    If no template is provided, a default one is generated based on available column names.

    Enhanced Template Features:
    1. Standard placeholders: "{text}", "{url}"
    2. Nested dictionary access using dot notation: "{metadata.author}", "{header.title}"
    3. Array indexing: "{questions[0].text}", "{options[2]}"
    4. Function calls: "{to_letter(score)}" - converts numbers to letters (A, B, C, ...)
       - Optional case parameter: "{to_letter(score, case='lower')}" for lowercase letters
       - Handles numbers > 25 with multiple letters (A-Z, then AA, AB, etc.)
    """

    def __init__(
        self,
        dataset_configs: List[Dict],
        # HF AutoTokenizer instance of string path and args
        tokenizer,
        tokenizer_args: Dict = {},
        # Distributed training parameters, default to 8 GPUs
        rank: int = 0,
        world_size: int = 8,
        # Packing parameter sizes
        packing_batch_size: int = 8,
        packing_context_length: int = 4096,
        packing_skip: bool = False, # Skip packing if True (used for debugging)
        
        # Shuffling parameters
        shuffle_buffer_size: int = 100,
        seed: int = 42, # Note that seed is incremented for each dataset "loop"

        # Interleaving stopping strategy
        # use either: 'first_exhausted', 'all_exhausted'
        interleaving_stopping_strategy: str = 'all_exhausted',

        # Multiprocessing context
        mp_context = None
    ):
        """
        Initialize the HFDatasetStreamer.

        Args:
            dataset_configs: A list of dictionaries, each containing dataset configuration with the following keys:
                - hf_path (required): The HuggingFace dataset path to load.
                - hf_args (optional): Arguments to pass to load_dataset function. Defaults for 'split' and 'streaming' 
                  are 'train' and True respectively if not provided.
                - template (optional): A string template for formatting dataset rows. Uses Python's string formatting 
                  syntax with named placeholders corresponding to dataset column names (e.g., "{text}", "{url}").
                  If not provided, a default template is generated based on available column names.
                - min_length (optional): Minimum token length for filtering. Default is 1.
                - max_length (optional): Maximum token length for filtering. Default is infinity.
                - weight (optional): Weight for this dataset when interleaving. Default is 1.0.
                
                Template Examples:
                - Basic text: "{text}"
                - With metadata: "Title: {title}\n\nContent: {content}"
                - With URL: "---\nurl: {url}\n---\n{text}"
                - Instruction format: "### Instruction:\n{instruction}\n\n### Response:\n{response}"
                - Nested dictionary access: "Author: {metadata.author}\nPublished: {metadata.date}"
                - Array indexing: "Question: {questions[0].text}\nOptions: {options[0]}, {options[1]}, {options[2]}"
                - Numeric to letter conversion: "Answer: {to_letter(answer_index)}" (0 → A, 1 → B, etc.)
                - Lowercase letters: "Choice: {to_letter(choice, case='lower')}" (0 → a, 1 → b, etc.)
                
            tokenizer: An AutoTokenizer instance or a string specifying the tokenizer name.
            tokenizer_args: Optional arguments to pass to the tokenizer. Used only when tokenizer is a string.

            rank: The rank of the current process.
            world_size: The total number of distributed processes.
            packing_batch_size: The number of sequences in each batch.
            packing_context_length: The maximum length of each sequence.
            packing_skip: Skip packing if True (used for debugging).

            shuffle_buffer_size: The size of the buffer used for shuffling.
            seed: The random seed used for shuffling.
            interleaving_stopping_strategy: The stopping strategy for interleaving datasets (default: 'all_exhausted').
            mp_context: Optional multiprocessing context to use. If None, uses the current context.
        """
        # Get or create multiprocessing context
        self.mp_context = mp_context or mp.get_context()

        # Prepare the tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_args) if isinstance(tokenizer, str) else tokenizer
            self.tokenizer = tokenizer
        except Exception as e:
            print("[ERROR] Failed to create tokenizer", e)
            raise e

        # Save the various settings
        self.dataset_configs = dataset_configs

        self.rank = rank
        self.world_size = world_size
        self.packing_batch_size = packing_batch_size
        self.packing_context_length = packing_context_length
        self.packing_skip = packing_skip

        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.interleaving_stopping_strategy = interleaving_stopping_strategy

        # Track the number of dataset reshuffle cycles
        self._reshuffle_count = 0

        # The internal tokenizer lock
        # to avoid multi-threaded tokenization issues
        self._tokenizer_lock = self.mp_context.Lock()

        # Initialize, and reshuffle the dataset
        self.reshuffle_dataset(0)

    def thread_safe_tokenizer_encode(self, text):
        # Single thread safe tokenization
        # While this is potenitally a bottleneck, HFDatasetStreamer
        # is meant to be an instance per GPU. So it should be "ok"
        try:
            with self._tokenizer_lock:
                # print("tokenizing text len:", len(text))
                return self.tokenizer.encode(text)
        except Exception as e:
            print("[ERROR] Failed to tokenize text", e)
            raise e

    def reshuffle_dataset(self, reshuffle_count=-1):
        # Build the shuffle seed. we also add a prime number that is large enough
        # to reduce some wierd not-so-random shuffling patterns, with small seeds
        if reshuffle_count == -1:
            self._reshuffle_count += 1
            shuffle_seed = self.seed + self._reshuffle_count + 17167 
        else:
            shuffle_seed = self.seed + reshuffle_count + 17167

        # Load and preprocess datasets
        multi_datasets = []
        multi_score_weight = []
        for ds_config in self.dataset_configs:
            # Get the hf_args, set split/streaming defaults
            hf_args = ds_config.get('hf_args', {})
            if 'split' not in hf_args:
                hf_args['split'] = 'train'
            if 'streaming' not in hf_args:
                hf_args['streaming'] = True

            # The hf_path to use
            hf_path = ds_config['hf_path']

            # Load the dataset
            one_dataset = load_dataset(
                hf_path,
                **hf_args
            )

            # Log that the dataset is being prepaird
            print(f"## Preparing dataset: {hf_path} ...")

            # Shard the dataset for distributed training
            one_dataset = one_dataset.shard(num_shards=self.world_size, index=self.rank)

            # Get and Apply the formatting template
            # The template is a string with Python's string formatting syntax using named placeholders
            # that correspond to dataset column names, e.g., "{text}", "{url}", "{instruction}", etc.
            ds_template = ds_config.get('template', None)
            if ds_template is None:
                    
                # Check the column names, if given
                # Otherwise sample the first row to get them
                ds_column_names = one_dataset.column_names
                if ds_column_names is None:
                    # Fetch the first row via iteration
                    # as it may not have index and length implemented
                    for row in one_dataset:
                        ds_column_names = list(row.keys())
                        break

                # Generate a default template based on the column names
                # Default template generation logic:
                # 1. If 'text' column exists, use it as the main content
                # 2. If 'url' column exists alongside 'text', add it as metadata
                ds_template = ''
                if 'text' in ds_column_names:
                    if 'url' in ds_column_names:
                        ds_template += '---\n'
                        ds_template += 'url: {url}\n'
                        ds_template += '---\n'
                    ds_template += '{text}'

            # Check if the template is empty, then raise an error
            # This happens if no template was provided and no 'text' column was found
            if ds_template == '':
                raise ValueError("Template is empty and no 'text' column found in dataset.")

            # Apply template function encapsulation
            # to avoid ds_template scope leakage
            def apply_template(in_dataset, in_dataset_path, in_template):
                def format_template(x):
                    try:
                        # Use the enhanced TemplateFormatter to format the template
                        # This supports:
                        # 1. Nested dictionary access with dot notation: {metadata.author}
                        # 2. Array indexing: {questions[0].text}
                        # 3. Function calls: {to_letter(score, case='upper')}
                        return TemplateFormatter.format_template(in_template, x)
                    except Exception as e:
                        # Handle formatting errors gracefully:
                        # - Missing keys in the dataset row
                        # - Type errors during formatting
                        # - Other formatting exceptions
                        # 
                        # Instead of failing, we log the error and return an empty string
                        # which will result in this row being filtered out later due to
                        # having zero tokens after tokenization
                        print("\n".join([
                            f"[WARNING] Error in formatting template, skipping data row",
                            f"- Template: {in_template}",
                            f"- Data Row: {x}",
                            f"- Exception: {e}",
                        ]))
                        return ''
                
                # Apply the formatting template to each row in the dataset
                # This transforms the dataset by:
                # 1. Applying the template to each row
                # 2. Creating a new 'text' field with the formatted result
                # 3. Adding 'hf_path' to track the source dataset
                # 4. Selecting only these two columns for further processing
                return in_dataset.map(
                    lambda x: { 'text': format_template(x), 'hf_path': in_dataset_path },
                    batched=False
                ).select_columns(['text', 'hf_path'])
            
            # Apply the template to the dataset
            one_dataset = apply_template(one_dataset, hf_path, ds_template)

            # Tokenize the text, via a single process
            one_dataset = one_dataset.map(
                lambda x: { 'input_ids': self.thread_safe_tokenizer_encode(x['text']) },
                batched=False
            )

            # Filter by token length if specified
            min_length = ds_config.get('min_length', 1)
            max_length = ds_config.get('max_length', float('inf'))
            one_dataset = one_dataset.filter(
                lambda x: (min_length <= len(x["input_ids"]) <= max_length)
            )

            # Inner shuffle of the dataset
            one_dataset = one_dataset.shuffle(seed=shuffle_seed, buffer_size=self.shuffle_buffer_size)

            # Append the dataset to the list
            multi_datasets.append(one_dataset)

            # Get the score weight
            score_weight = ds_config.get('weight', 1.0)
            multi_score_weight.append(float(score_weight))

        # Normalize the weights (sum to 1)
        weight_sum = sum(multi_score_weight)
        if weight_sum > 0:  # Avoid division by zero
            multi_score_weight = [w / weight_sum for w in multi_score_weight]
        else:
            # If all weights are zero, use equal weights
            multi_score_weight = [1.0 / len(multi_score_weight) for _ in multi_score_weight]

        # Due to FP accuracy issues, one of the weights may be slightly off, and not add up to 1
        # This is to fix that, by rounding the last weight to `1 - sum(all other weights)`
        # After rounding down all numbers to 3 decimal places
        if sum(multi_score_weight) != 1.0:
            multi_score_weight = [math.floor(w*1000)/1000.0 for w in multi_score_weight]
            multi_score_weight[-1] = 1.0 - sum(multi_score_weight[:-1])

        # print the interleaving step
        print(f"## Interleaving datasets with weights: {multi_score_weight}")

        # Interleave datasets if multiple are provided
        full_dataset = interleave_datasets(
            multi_datasets, seed=shuffle_seed, 
            probabilities=multi_score_weight, 
            stopping_strategy=self.interleaving_stopping_strategy
        )
        
        # Outer shuffle of the dataset
        print(f"## Post interleaving shuffle ...")
        full_dataset = full_dataset.shuffle(seed=shuffle_seed+1, buffer_size=100)

        print("## Dataset reshuffled. Ready for streaming.")

        # Register the full dataset
        self.full_dataset = full_dataset

    def __iter__(self):
        """
        Iterate over the dataset, yielding packed batches of tokenized sequences.

        Yields:
            torch.Tensor: A batch of tokenized sequences, shape (packing_batch_size, packing_context_length).
        """
        iterator = iter(self.full_dataset)

        # No packing, just yield the samples
        # ------------------------
        if self.packing_skip:
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    # Reshuffle the dataset and create a new iterator
                    self.reshuffle_dataset()
                    iterator = iter(self.full_dataset)

        # Time to do some packing
        # ------------------------
        buffer = []
        current_length = 0

        # Full target buffer size
        target_buffer_size = self.packing_context_length * self.packing_batch_size

        while True:
            try:
                # The list of, text and sources used
                text_arr = []
                sources = []

                # Fill the buffer until it has enough tokens for a full batch
                while current_length < target_buffer_size:
                    sample = next(iterator)

                    text_arr.append(sample["text"])
                    sources.append(sample["hf_path"])

                    buffer.extend(sample["input_ids"] + [self.tokenizer.eos_token_id])
                    current_length += len(sample["input_ids"]) + 1

                # Extract a full batch from the buffer
                batch = buffer[:self.packing_context_length * self.packing_batch_size]
                buffer = buffer[self.packing_context_length * self.packing_batch_size:]
                current_length = len(buffer)

                # Yield the batch as a tensor
                yield {
                    "input_ids": torch.tensor(batch, dtype=torch.long).reshape(self.packing_batch_size, self.packing_context_length),
                    "text_arr": text_arr,
                    "source_arr": sources
                }

            except StopIteration:
                # If the iterator is exhausted, skip the current batch, and reset the buffer
                # And reshuffle the dataset and create a new iterator
                self.reshuffle_dataset()
                iterator = iter(self.full_dataset)
                buffer = []
                current_length = 0

    def __len__(self):
        raise NotImplementedError("HFDatasetStreamer does not support __len__.")
    
    def __getitem__(self, index):
        raise NotImplementedError("HFDatasetStreamer does not support __getitem__.")
