"""Command-line demonstration for Assignment 1B."""

import argparse

from src.model import DEFAULT_MODEL_ID, get_response, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat with an open-weight instruct model on local hardware."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--seed", type=int, default=6, help="Reproducible sampling seed")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--prompt",
        action="append",
        help="Run one prompt non-interactively; repeat for several prompts",
    )
    return parser.parse_args()


def answer(prompt, model, tokenizer, args):
    print(f"\nYou: {prompt}")
    response = get_response(
        prompt,
        model,
        tokenizer,
        max_length=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(f"Model: {response}\n")


def main():
    args = parse_args()
    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model)
    parameter = next(model.parameters())
    print(f"Ready | device={model.device} | dtype={parameter.dtype}")

    if args.prompt:
        for prompt in args.prompt:
            answer(prompt, model, tokenizer, args)
        return

    print("Enter a question, or type 'quit' to exit.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if prompt.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not prompt:
            continue
        answer(prompt, model, tokenizer, args)


if __name__ == "__main__":
    main()
