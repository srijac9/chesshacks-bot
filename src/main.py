import os
from .utils import chess_manager, GameContext
from chess import Move
import random
import threading

# Transformers & PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import re


# =========================
# Global Variables
# =========================
model = None
tokenizer = None
model_ready = False


# =========================
# Lazy Model Loading
# =========================
def load_model():
    global model, tokenizer, model_ready

    try:
        print("Loading base model...")

        BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
        LORA_REPO = "jasmehar/qwen-chess-gm-lora"  

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        # Load base model on GPU
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print("Loading LoRA adapter from HuggingFace...")

        model = PeftModel.from_pretrained(
            base_model,
            LORA_REPO
        )

        model.eval()

        print("Model initialization complete! (Base + LoRA)")
        model_ready = True

    except Exception as e:
        print("\n=== MODEL LOAD ERROR ===")
        print(e)
        print("========================\n")

# Start model loading thread
threading.Thread(target=load_model, daemon=True).start()



# =========================
# Bot Move Function
# =========================
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    global model_ready

    if not model_ready:
        print("model not ready")
        return random.choice(list(ctx.board.generate_legal_moves()))

    print("model ready: generate move")

    fen = ctx.board.fen()

    # List all legal moves
    legal_moves = [m.uci() for m in ctx.board.generate_legal_moves()]
    moves_str = ", ".join(legal_moves)

    messages = [
        {
            "role": "system",
            "content":
                "You are a STRICT chess engine. Output ONLY one legal move.\n"
                "MOVE MUST be in a-h 1-8 range.\n"
                "Legal moves:\n"
                f"{moves_str}\n"
                "Only output the move in UCI."
        },
        {"role": "user", "content": f"FEN: {fen}\nMove:"}
    ]

    # Build prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0
        )

    # Extract new tokens
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = out[0][prompt_len:]
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print("\n=== MODEL RAW OUTPUT ===")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    print("=== COMPLETION ONLY ===")
    print(completion)
    print("========================\n")

    # Extract UCI move
    match = re.search(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", completion.lower())

    if not match:
        print("Could not find valid UCI in:", completion)
        return random.choice(list(ctx.board.generate_legal_moves()))

    predicted = match.group(0)
    print("Predicted UCI:", predicted)

    # Validate move
    try:
        move = Move.from_uci(predicted)
    except Exception:
        print("Move.from_uci failed:", predicted)
        return random.choice(list(ctx.board.generate_legal_moves()))

    if move not in ctx.board.generate_legal_moves():
        print("Illegal move:", predicted)
        return random.choice(list(ctx.board.generate_legal_moves()))

    print("legal")
    ctx.logProbabilities({})
    return move



# =========================
# Reset Function
# =========================
@chess_manager.reset
def reset_func(ctx: GameContext):
    pass



# =========================
# Server Start
# =========================
if __name__ == "__main__":
    print("Starting chess engine server...")
    chess_manager.serve()
