import torch
import matplotlib.pyplot as plt
from src.model import WordGuesser
from src.utils import one_hot_word, cross_entropy_loss

TARGET_WORD = "CRANE"
EPOCHS = 100

model = WordGuesser()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
target = one_hot_word(TARGET_WORD)

loss_history = []

for step in range(EPOCHS):
    optimizer.zero_grad()
    pred_probs = model()
    loss = cross_entropy_loss(pred_probs, target)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    guessed_letters = [chr(p.argmax().item() + ord('A')) for p in pred_probs]
    print(f"Step {step:3d} | Loss: {loss.item():.4f} | Guess: {''.join(guessed_letters)}")

# Plotting loss
plt.plot(loss_history)
plt.xlabel("Step")
plt.ylabel("Cross Entropy Loss")
plt.title("Loss over Time")
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()
