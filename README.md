# Cross Entropy Word Guesser 🔤🎯

An experimental project that turns the concept of cross entropy into a gamified word-guessing AI — think Wordle meets Information Theory.

## 🔍 What It Does
- Learns to guess a hidden 5-letter word (like "CRANE")
- Uses a neural net to output letter probabilities at each position
- Cross entropy is used as the loss function to train it toward the correct word

## 💡 Why This Is Cool
- Demonstrates entropy minimization visually
- Connects ML concepts to games and natural language
- Can be extended into multi-word training or transformer-based language guessing

## 🧠 Future Ideas
- Add user-input mode
- Integrate Gradio/Streamlit UI
- Compare with GPT token predictions

## 🚀 Run It
```bash
pip install -r requirements.txt
python main.py
