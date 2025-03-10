from datasets import load_dataset
import datasets
from transformers import AutoTokenizer
import concurrent.futures
import os, math, torch

def full_build_mmlu_test_dataset(
    n_shot=0,
    # Tokenizer can be the AutoTokenizer instance
    # or a string, which is used to load the tokenizer
    tokenizer="neox",
    # Minimum right padding length
    min_right_pad_tokens=0,
    # prompt chunk size (in tokens)
    prompt_chunk_size=16,
    # Use validation set instead of test set
    use_validation_set=False
):
    """
    Build the MMLU test dataset, with the given tokenizer and paddings
    """
    # Single processing mode (avoid some known issues, with world tokenizer)
    single_process_mode = False

    # Build the tokenizer
    tokenizer_str = None
    if isinstance(tokenizer, str):
        if tokenizer == "world":
            tokenizer_str = "world"
            tokenizer = AutoTokenizer.from_pretrained("RWKV/v6-Finch-1B6-HF", trust_remote_code=True)
            single_process_mode = True
        elif tokenizer == "neox":
            tokenizer_str = "neox"
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Aliasing tokenizer class name to short str
    if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
        tokenizer_str = "neox"
    elif tokenizer.__class__.__name__ == "RwkvTokenizer":
        tokenizer_str = "world"
        single_process_mode = True

    ## HOTFIX: Set single process mode, for much better dataset building reliability
    ## the minimal time saving is not substential enough to justify tracing and fixing the issues
    single_process_mode = True

    # Check if its based on AutoTokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer shuld be an instance of AutoTokenizer or a string (which is used to load the AutoTokenizer).")
    
    # HF dataset to use
    hf_dataset_name = "cais/mmlu"

    # Dataset threads to use
    dataset_threads = max(int(os.cpu_count()), 1)

    # test dataset map
    test_dataset_map = {}

    # Choice letters
    choice_letters = ["A", "B", "C", "D"]
    choice_tokens = [tokenizer.encode(letter, return_tensors="pt", padding=False)[0][0] for letter in choice_letters]

    # The list of dataset subset
    subject_list = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "elementary_mathematics",
        "electrical_engineering",
        "formal_logic",
        "global_facts",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_law",
        "professional_accounting",
        "professional_psychology",
        "professional_medicine",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
        # ---
        # "humanities",
        # "other",
        # "stem",
        # "social_sciences",
    ]

    # Set name to use (test or val)
    if use_validation_set:
        test_set = "validation"
        subject_list = ["all"]
    else:
        test_set = "test"

    # Subject threads
    subject_threads = max( math.ceil(dataset_threads/len(subject_list)), 2 )
    if single_process_mode:
        subject_threads = 1

    # Build the cache key
    format_revision = 0
    cache_key = f"mmlu-{test_set}-t_{tokenizer_str}-n_{n_shot}-p_{min_right_pad_tokens}-c_{prompt_chunk_size}-r{format_revision}"

    # Hash of the current file
    script_file_path = __file__
    script_file_content = open(script_file_path, "rb").read()
    script_file_hash = hash(script_file_content)

    # Disable progress bars, as they are noisy in logs
    datasets.disable_progress_bars()
    # # Disable caching as it has known issues
    # datasets.disable_caching()

    # Fromat data sample, from question, and choices, and the answer
    def format_datasample(datasample, include_answer=True):
        res = "" + datasample["question"]
        for i, choice in enumerate(datasample["choices"]):
            res += f"\n{choice_letters[i]}. {choice}"
        res += "\nAnswer: "
        if include_answer:
            res += f"{choice_letters[datasample["answer"]]}\n\n"
        return res

    # Get the few shot prompt prefix
    def get_fewshot_examples(subject):
        if n_shot == 0:
            return ""
        
        # Dev dataset is apperantly used for few shot 
        dev_dataset = load_dataset(hf_dataset_name, subject, split="dev")

        # Build and return the prompt
        ret_prompt = ""
        for i in range(n_shot):
            ret_prompt += format_datasample(dev_dataset[i], include_answer=True)
        return ret_prompt

    def process_subject(subject):
        prompt_prefix = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            subject.replace("_", " ")
        )

        print("## Building dataset for {} subject (n_shot={}): {}".format(test_set, n_shot,subject))

        # Prepare the few shot example, if needed
        prompt_prefix += get_fewshot_examples(subject)

        # dataset_formatter
        def dataset_formatter(datasample):
            prompt = prompt_prefix + format_datasample(datasample, include_answer=False)

            prompt_input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=False)[0].to("cpu")
            prompt_length = prompt_input_ids.shape[0]
            return {
                "prompt": prompt_input_ids,
                "prompt_length": prompt_length,
                "answer": datasample["answer"]
            }

        # Build the dataset
        test_dataset = load_dataset(hf_dataset_name, subject, split=test_set)
        formatted_dataset = test_dataset.map(dataset_formatter, batched=False, num_proc=subject_threads, new_fingerprint=str(hash(f"{script_file_hash}-{cache_key}-{subject}-formatter")))

        # Get the longest prompt length
        subject_longest_prompt_length = 0
        for i in range(len(formatted_dataset)):
            subject_longest_prompt_length = max(subject_longest_prompt_length, formatted_dataset[i]["prompt_length"])

        # Return the subject, and the formatted dataset, and the longest prompt length
        return subject, formatted_dataset, subject_longest_prompt_length

    # Process the subjects, in parallel
    longest_prompt_length = 0
    if single_process_mode:
        for subject in subject_list:
            subject, processed_dataset, subject_longest_prompt_length = process_subject(subject)
            test_dataset_map[subject] = processed_dataset
            longest_prompt_length = max(longest_prompt_length, subject_longest_prompt_length)
            print("## Dataset is ready for {} subject (n_shot={}): {}".format(test_set, n_shot,subject))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=dataset_threads) as executor:
            futures = {executor.submit(process_subject, subject): subject for subject in subject_list}
            for future in concurrent.futures.as_completed(futures):
                subject, processed_dataset, subject_longest_prompt_length = future.result()
                test_dataset_map[subject] = processed_dataset
                longest_prompt_length = max(longest_prompt_length, subject_longest_prompt_length)
                print("## Dataset is ready for {} subject (n_shot={}): {}".format(test_set, n_shot,subject))

    # Longest prompt token length
    print("## Longest prompt token length:", longest_prompt_length)

    # Calculate the target prompt+padding length
    target_prompt_length = longest_prompt_length + min_right_pad_tokens
    target_prompt_length = math.ceil(target_prompt_length / prompt_chunk_size) * prompt_chunk_size
    print("## Padding to target prompt length:", target_prompt_length)

    # Pad the prompt
    def pad_prompt_tokens(prompt):
        # If prompt is a list, convert it to tensor
        if isinstance(prompt, list):
            prompt = torch.tensor(prompt, dtype=torch.long, device="cpu")

        # Skip if prompt is already padded
        if prompt.shape[0] >= target_prompt_length:
            return prompt
        
        # Pad the prompt 
        return torch.cat(
            [
                prompt,
                torch.zeros(target_prompt_length - prompt.shape[0], dtype=prompt.dtype, device="cpu"),
            ],
            dim=0,
        )

    # Format the datasample
    def pad_datasample(datasample):
        return {
            "prompt": pad_prompt_tokens(datasample["prompt"]), 
            "prompt_length": datasample["prompt_length"],
            "answer": datasample["answer"],
        }

    def pad_and_finalize_subject(subject):
        # Pad the prompt
        padded_dataset = test_dataset_map[subject].map(pad_datasample, batched=False, num_proc=subject_threads, new_fingerprint=str(hash(f"{script_file_hash}-{cache_key}-{subject}-pad-n-finalize")))

        # Finalize the dataset, as a single tensor per subject
        dataset_length = len(padded_dataset)
        final_dataset = {
            "prompt_id": torch.zeros(dataset_length, target_prompt_length, dtype=torch.long),
            "prompt_length": torch.zeros(dataset_length, dtype=torch.long),
            "answer_id": torch.zeros(dataset_length, dtype=torch.long),
        }
        for i in range(dataset_length):
            final_dataset["prompt_id"][i] = torch.tensor(padded_dataset[i]["prompt"], dtype=torch.long)
            final_dataset["answer_id"][i] = padded_dataset[i]["answer"]
            final_dataset["prompt_length"][i] = padded_dataset[i]["prompt_length"]

        # Return the subject, and the final dataset
        return subject, final_dataset

    # Pad the subjects in parallel
    if single_process_mode:
        for subject in subject_list:
            subject, updated_dataset = pad_and_finalize_subject(subject)
            test_dataset_map[subject] = updated_dataset
            print("## Dataset is padded for {} subject (n_shot={}): {}".format(test_set, n_shot,subject))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=dataset_threads) as executor:
            futures = {executor.submit(pad_and_finalize_subject, subject): subject for subject in subject_list}
            for future in concurrent.futures.as_completed(futures):
                subject, updated_dataset = future.result()
                test_dataset_map[subject] = updated_dataset
                print("## Dataset is padded for {} subject (n_shot={}): {}".format(test_set, n_shot,subject))

    # Save the answer token IDs
    test_dataset_map["_answer_token_id"] = torch.tensor(choice_tokens, dtype=torch.long)

    # Return the test dataset map
    return test_dataset_map

def cache_build_mmlu_test_dataset(
    n_shot=0,
    # Cachable test dataset, needs a tokenizer str
    tokenizer="neox",
    # Minimum right padding length
    # of blank tokens, to the right
    min_right_pad_tokens=0,
    # prompt chunk size (in tokens)
    # to round up in chunks of
    prompt_chunk_size=16,
    # Cache directory
    test_cache_dir=-1,
    # Use validation set instead of test set
    use_validation_set=False,
    # Read only from cache (no build)
    read_only_cache=False
):
    """
    Build the MMLU test dataset, with the given tokenizer and paddings
    And cache it, if possible
    """

    # Tokenizer class name is used if not a string
    if not isinstance(tokenizer, str):
        tokenizer_str = tokenizer.__class__.__name__

        # Aliasing tokenizer class name to short str
        if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
            tokenizer_str = "neox"
        elif tokenizer.__class__.__name__ == "RwkvTokenizer":
            tokenizer_str = "world"
    else:
        tokenizer_str = tokenizer
    
    # Set name to use (test or val)
    test_set = "val" if use_validation_set else "test"

    # Build the cache key
    format_revision = 0
    cache_key = f"mmlu-{test_set}-t_{tokenizer_str}-n_{n_shot}-p_{min_right_pad_tokens}-c_{prompt_chunk_size}-r{format_revision}.pth"

    # Use the __file__/.mmlu_cache as the cache directory
    if test_cache_dir == -1:
        test_cache_dir = os.path.join(os.path.dirname(__file__), ".mmlu_cache")

    # Check if the cache exists
    cache_path = os.path.join(test_cache_dir, cache_key)
    if os.path.exists(cache_path):
        print(f"## Loading MMLU cached dataset (n_shot={n_shot},tokenizer={tokenizer_str}):", cache_path)
        return torch.load(cache_path, weights_only=True, mmap=True, map_location="cpu")
        # with open(cache_path, "rb") as f:
        #     return pickle.load(f)
    else:
        print(f"## Building MMLU cached dataset (n_shot={n_shot},tokenizer={tokenizer_str}):", cache_path)
    
    # If read only cache, return None
    if read_only_cache:
        return None

    # Build the dataset
    built_dataset = full_build_mmlu_test_dataset(
        n_shot=n_shot,
        tokenizer=tokenizer,
        min_right_pad_tokens=min_right_pad_tokens,
        prompt_chunk_size=prompt_chunk_size,
        use_validation_set=use_validation_set
    )

    # Save the dataset
    print(f"## Saving MMLU dataset cache (n_shot={n_shot},tokenizer={tokenizer_str}):", cache_path)
    os.makedirs(test_cache_dir, exist_ok=True)
    torch.save(built_dataset, cache_path)
    # with open(cache_path, "wb") as f:
    #     pickle.dump(built_dataset, f)

    # Return the dataset
    return built_dataset

def get_built_mmlu_test_dataset(
    n_shot=0,
    # Cachable test dataset, needs a tokenizer str
    tokenizer="neox",
    # Minimum right padding length
    # of blank tokens, to the right
    min_right_pad_tokens=0,
    # prompt chunk size (in tokens)
    # to round up in chunks of
    prompt_chunk_size=16,
    # Cache directory
    test_cache_dir=-1,
    # Use validation set instead of test set
    use_validation_set=False
):
    # Get the cached dataset, if its built, or throw an error with the CLI command
    ret = cache_build_mmlu_test_dataset(
        n_shot=n_shot,
        tokenizer=tokenizer,
        min_right_pad_tokens=min_right_pad_tokens,
        prompt_chunk_size=prompt_chunk_size,
        test_cache_dir=test_cache_dir,
        use_validation_set=use_validation_set,
        read_only_cache=True
    )
    if ret is None:
        print("## MMLU test dataset not built, run the following command to build it:")
        print(f"python {__file__} --n_shot {n_shot} --tokenizer {tokenizer} --use_validation_set {use_validation_set} --min_right_pad_tokens {min_right_pad_tokens} --prompt_chunk_size {prompt_chunk_size}")
        raise RuntimeError("MMLU test dataset not built")
    return ret

if __name__ == "__main__":
    import argparse
    def main():
        parser = argparse.ArgumentParser(description="Build and cache MMLU test dataset")
        parser.add_argument("--n_shot", type=int, default=0, help="Number of few-shot examples")
        parser.add_argument("--tokenizer", type=str, default="neox", help="Tokenizer to use")
        parser.add_argument("--hf_model", type=str, default=None, help="Hugging Face model path to use for AutoTokenizer")
        parser.add_argument("--use_validation_set", action="store_true", help="Use validation set instead of test set")
        parser.add_argument("--min_right_pad_tokens", type=int, default=0, help="Minimum right padding length")
        parser.add_argument("--prompt_chunk_size", type=int, default=16, help="Prompt chunk size in tokens")
        parser.add_argument("--test_cache_dir", type=str, default="-1", help="Cache directory (default to `__dirname__/.mmlu_cache`)")

        args = parser.parse_args()

        tokenizer = args.tokenizer
        if args.hf_model:
            print("## Using HF model tokenizer:", args.hf_model)
            tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
        else:
            print("## Using tokenizer:", tokenizer)

        cache_build_mmlu_test_dataset(
            n_shot=args.n_shot,
            tokenizer=tokenizer,
            use_validation_set=args.use_validation_set,
            min_right_pad_tokens=args.min_right_pad_tokens,
            prompt_chunk_size=args.prompt_chunk_size,
            test_cache_dir=args.test_cache_dir if args.test_cache_dir != "-1" else -1
        )
        print("## Done: Dataset has been built and cached")
    main()
