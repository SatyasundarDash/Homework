import numpy as np

NAMES = ["person", "chair", "book"]
OUTPUT = np.array([
    [0.90, 0.90, 0.05, 0.05],
    [0.80, 0.10, 0.80, 0.10],
    [0.60, 0.20, 0.10, 0.70],
    [0.50, 0.60, 0.30, 0.10],
    [0.10, 0.90, 0.05, 0.05],
    [0.70, 0.30, 0.20, 0.90],
])
MIN_SCORE = 0.5

# ---------- PART 1: score every guess ----------
scores = OUTPUT[:, 0:1] * OUTPUT[:, 1:4]
print(scores.shape)

print("PART 1: Scores")
print("Shape:", scores.shape)
print("Guess 4, person:", round(float(scores[4, 0]), 2))

# ---------- PART 2: the best name for every guess ----------
# class_ids = the position of the biggest score in each row. confidences = that biggest score.
class_ids = np.argmax(scores, axis=1) 
confidences = np.max(scores, axis=1)    

print("\nPART 2: Best name")
for i in range(6):
    print("Guess", i, "->", NAMES[class_ids[i]], f"{confidences[i]:.2f}")

# ---------- PART 3: keep the sure guesses ----------
# kept = the numbers of the guesses whose confidence is at least MIN_SCORE.
kept = np.where(confidences >= MIN_SCORE)[0]
print("\nPART 3: Kept")
print("Kept guesses:", [int(i) for i in kept])
print("Kept names:", [NAMES[class_ids[i]] for i in kept])

# ---------- PART 4: change how sure it must be ----------
print("\nPART 4: Three thresholds")
for threshold in [0.2, 0.5, 0.8]:
    count = np.sum(confidences >= threshold)
    print(f"Threshold {threshold} -> {count} kept")
    
# Threshold <threshold> -> <number> kept
