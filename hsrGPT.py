from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
import os
import random

# one-hot encoding
def one_hot_encode_cards(X_raw):
    N = X_raw.shape[0]
    X_encoded = np.zeros((N, 85))
    for i in range(5):
        suit = np.clip(X_raw[:, i * 2].astype(int), 0, 3)
        rank = np.clip(X_raw[:, i * 2 + 1].astype(int), 1, 13)
        suit_one_hot = np.zeros((N, 4))
        suit_one_hot[np.arange(N), suit] = 1
        rank_one_hot = np.zeros((N, 13))
        rank_one_hot[np.arange(N), rank - 1] = 1
        X_encoded[:, i * 17 : i * 17 + 4] = suit_one_hot
        X_encoded[:, i * 17 + 4 : (i + 1) * 17] = rank_one_hot
    return X_encoded

def generate_hand(class_id):
    suits = [0, 1, 2, 3]
    ranks = list(range(1, 14))

    if class_id == 9:  # Royal Flush
        suit = random.choice(suits)
        return [suit, 10, suit, 11, suit, 12, suit, 13, suit, 1]

    elif class_id == 8:  # Straight Flush
        suit = random.choice(suits)
        start = random.choice(range(1, 10))  # A-5 to 9-K
        return [suit, start, suit, start+1, suit, start+2, suit, start+3, suit, start+4]

    elif class_id == 7:  # Four of a Kind
        rank = random.choice(ranks)
        suits_for_quad = random.sample(suits, 4)
        fifth_card = random.choice([(s, r) for s in suits for r in ranks if r != rank])
        return suits_for_quad[0:1] + [rank] + suits_for_quad[1:2] + [rank] + suits_for_quad[2:3] + [rank] + suits_for_quad[3:4] + [rank] + [fifth_card[0], fifth_card[1]]

    elif class_id == 6:  # Full House
        triple_rank, pair_rank = random.sample(ranks, 2)
        triple_suits = random.sample(suits, 3)
        pair_suits = random.sample(suits, 2)
        return triple_suits[0:1] + [triple_rank] + triple_suits[1:2] + [triple_rank] + triple_suits[2:3] + [triple_rank] + pair_suits[0:1] + [pair_rank] + pair_suits[1:2] + [pair_rank]

    elif class_id == 5:  # Flush
        suit = random.choice(suits)
        chosen_ranks = random.sample(ranks, 5)
        return sum([[suit, r] for r in chosen_ranks], [])

    elif class_id == 4:  # Straight
        start = random.choice(range(1, 10))  # A-5 to 9-K
        chosen_suits = random.choices(suits, k=5)
        return sum([[chosen_suits[i], start + i] for i in range(5)], [])

    else:
        raise ValueError("Only supports class 4-9")


def generate_synthetic_data(num_per_class=25000):
    synthetic_hands = []
    labels = []
    for class_id in range(4, 10):  # Rare classes only
        for _ in range(num_per_class):
            hand = generate_hand(class_id)
            synthetic_hands.append(hand)
            labels.append(class_id)
    return np.array(synthetic_hands), np.array(labels)


# load data
data = pd.read_csv('/Users/yayaj/Desktop/Java - VSC/AI/FinalProj/Training Sets/poker-hand-training-true.data', header=None)
X_raw = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X = one_hot_encode_cards(X_raw)
y_cat = to_categorical(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# generate and encode synthetic hands
X_synthetic_raw, y_synthetic = generate_synthetic_data(25000)
X_synthetic = one_hot_encode_cards(X_synthetic_raw)
y_synthetic_categorical = to_categorical(y_synthetic, num_classes=10)

# merge syn with og training set
X_train_augmented = np.vstack([X_train, X_synthetic])
y_train_augmented = np.vstack([y_train, y_synthetic_categorical])


# class weights to balance the training
y_integers = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weights_dict = dict(enumerate(class_weights))

# build model
model = Sequential([
    Dense(512, activation='relu', input_shape=(85,)),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_augmented, y_train_augmented, epochs=50, batch_size=64, validation_split=0.1, class_weight=class_weights_dict, verbose=1)

# save model
model.save("/Users/yayaj/Desktop/Java - VSC/AI/FinalProj/Saved Things/pokerhandV8_syn.keras")
print("✅ Model saved as pokerhandV8_syn.keras")
