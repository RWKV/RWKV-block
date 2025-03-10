import concurrent.futures
import os, math, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# https://stackoverflow.com/a/28151907
if __name__ == "__main__":
    from BuildTestMMLU import cache_build_mmlu_test_dataset
else:
    from .BuildTestMMLU import cache_build_mmlu_test_dataset

def mmlu_test_runner(
    forward_func,
    batch_size=16,
    mmlu_dataset=None
):
    '''
    Given the CasualLM forward function, run the MMLU test
    The forward function is expected to take in a tensor of input_id's (batch, seq)
    And return a tensor of output logits (batch, seq, vocab_size)
    '''

    # If MMLU dataset is missing, use the pile based, validation dataset
    if mmlu_dataset is None:
        from .BuildTestMMLU import cache_build_mmlu_test_dataset
        mmlu_dataset = cache_build_mmlu_test_dataset(use_validation_set=True)

    # Log the MMLU test run
    print("## Starting the MMLU test ...")

    # Get the choices labels
    answer_token_id = mmlu_dataset["_answer_token_id"]

    # Loop through the subject keys (non _ values)
    subject_keys = [k for k in mmlu_dataset.keys() if not k.startswith("_")]

    # Return result map
    result_map = {}

    # Overall accuracy and probability
    overall_accuracy_list = []
    overall_probability_list = []

    # Loop through the subjects
    for subject in subject_keys:
        # Get the subject data
        subject_data = mmlu_dataset[subject]

        # Get the number of batches
        subject_data_count = subject_data["prompt_id"].shape[0]
        num_batches = math.ceil(subject_data_count / batch_size)

        # Log the subject
        print(f"### Running MMLU test : {subject} (count={subject_data_count}, batches={num_batches}) ...")

        # list of answer softmax probability of the correct answer
        # list of answer match, after argmax (1 if correct, 0 if incorrect)
        answer_prob_list = []
        answer_match_list = []

        # Loop through the batches
        for batch_index in range(num_batches):
            # start and endin position
            start = batch_index * batch_size
            endin = min(start + batch_size, subject_data_count)

            # The data has be pre-tokenized, padded, and tensorized, as a giant tensor (Batch,Seq)
            prompt_id = subject_data["prompt_id"][start:endin]
            prompt_length = subject_data["prompt_length"][start:endin]
            answer_id = subject_data["answer_id"][start:endin]
            sub_batch_size = prompt_id.shape[0]

            # Forward the prompt_id tokens, and get the logits
            full_logits = forward_func(prompt_id)

            # Iterate the individual question answers
            for i in range(sub_batch_size):
                # Get the logits for the answer
                answer_logits = full_logits[i, prompt_length[i] - 1]

                # Get the softmax probability of the correct answer
                answer_prob = torch.nn.functional.softmax(torch.tensor([
                    answer_logits[answer_token_id[0]],
                    answer_logits[answer_token_id[1]],
                    answer_logits[answer_token_id[2]],
                    answer_logits[answer_token_id[3]]
                ], dtype=torch.float), dim=0)

                # Append the answer probability
                answer_prob_list.append(answer_prob[answer_id[i]].item())
                # Append the answer match
                if (int(torch.argmax(answer_prob).item()) == int(answer_id[i].item())):
                    answer_match_list.append(1)
                else:
                    answer_match_list.append(0)
    
        # Calculate the MMLU score for the subject
        answer_accuracy = sum(answer_match_list) / len(answer_match_list)
        answer_probability = sum(answer_prob_list) / len(answer_prob_list)

        # Log the subject result
        print(f"#### {subject} - accuracy={answer_accuracy:.4f} , probability={answer_probability:.4f}")

        # Add the subject result to the result map
        result_map[subject] = {
            "accuracy": answer_accuracy,
            "probability": answer_probability
        }
        overall_accuracy_list.append(answer_accuracy)
        overall_probability_list.append(answer_probability)

    # Compute the overall accuracy and probability
    overall_accuracy = sum(overall_accuracy_list) / len(overall_accuracy_list)
    overall_probability = sum(overall_probability_list) / len(overall_probability_list)

    # Save it to "all" key
    result_map["all"] = {
        "accuracy": overall_accuracy,
        "probability": overall_probability
    }
    # Log the overall result
    print("------------------------------------------------")
    print(f"### MMLU overall test result : accuracy={overall_accuracy:.4f} , probability={overall_probability:.4f}")
    print("------------------------------------------------")

if __name__ == "__main__":
    
    import argparse
    def main():
        parser = argparse.ArgumentParser(description="Build and cache MMLU test dataset")
        parser.add_argument("hf_model", type=str, help="Hugging Face model path to use for AutoTokenizer and AutoModelForCausalLM")
        parser.add_argument("--n_shot", type=int, default=0, help="Number of few-shot examples")
        parser.add_argument("--use_validation_set", action="store_true", help="Use validation set instead of test set")
        parser.add_argument("--min_right_pad_tokens", type=int, default=0, help="Minimum right padding length")
        parser.add_argument("--prompt_chunk_size", type=int, default=16, help="Prompt chunk size in tokens")
        parser.add_argument("--test_cache_dir", type=str, default="-1", help="Cache directory (default to `__dirname__/.mmlu_cache`)")
        parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: 'cuda')")
        parser.add_argument("--tmix_backend", type=str, default="auto", help="Backend for tensor mixing (default: 'auto')")
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing (default: 16)")

        args = parser.parse_args()

        print("------------------------------------------------")
        print("## Loading HF model:", args.hf_model)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model, trust_remote_code=True, tmix_backend=args.tmix_backend, device=args.device).to(args.device)

        print("------------------------------------------------")
        print("## Preparing the dataset")
        mmlu_dataset=cache_build_mmlu_test_dataset(
            n_shot=args.n_shot,
            tokenizer=tokenizer,
            use_validation_set=args.use_validation_set,
            min_right_pad_tokens=args.min_right_pad_tokens,
            prompt_chunk_size=args.prompt_chunk_size,
            test_cache_dir=args.test_cache_dir if args.test_cache_dir != "-1" else -1
        )
        print("## Done: Dataset has been built and cached")
        print("------------------------------------------------")

        def model_forward_for_logits(x):
            with torch.no_grad():
                return model(x).logits

        mmlu_test_runner(
            forward_func=model_forward_for_logits,
            batch_size=args.batch_size,
            mmlu_dataset=mmlu_dataset
        )
        
    main()
