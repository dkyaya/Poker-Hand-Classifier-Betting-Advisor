import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("/Users/yayaj/Desktop/Java - VSC/AI/FinalProj/Saved Things/pokerhandV7_syn.keras")

# Card options
suits = ['Hearts', 'Spades', 'Diamonds', 'Clubs', 'Unknown']
ranks = list(range(1, 14)) + ['Unknown']
hand_labels = [
    "Nothing in hand", "One pair", "Two pairs", "Three of a kind", "Straight",
    "Flush", "Full house", "Four of a kind", "Straight flush", "Royal flush"
]

# One-hot encoding for a single card
def encode_card(suit, rank):
    suit_vector = [0] * 4
    rank_vector = [0] * 13

    if suit in suits[:-1] and rank != 'Unknown':
        suit_vector[suits.index(suit)] = 1
        rank_vector[int(rank) - 1] = 1

    return suit_vector + rank_vector  # total 17

# Encode up to 5 cards, using zero vectors for "Unknown"
def encode_hand(card_inputs):
    encoded = []
    for suit, rank in card_inputs:
        encoded.extend(encode_card(suit, rank))
    # Fill missing cards with zero vectors
    while len(encoded) < 85:
        encoded.extend([0] * 17)
    return np.array([encoded])

# Simple heuristic advice (could be expanded)
def generate_advice(predicted_class, known_cards):
    known_count = sum(1 for s, r in known_cards if s != "Unknown" and r != "Unknown")
    label = hand_labels[predicted_class]
    if known_count < 5:
        return f"🕵️ Partial Hand — predicted: **{label}**. This is preliminary; final hand may change."
    elif predicted_class >= 5:
        return f"💪 Strong hand! Play aggressively with a **{label}**."
    elif predicted_class >= 3:
        return f"🙂 Decent hand. Consider betting modestly — **{label}**."
    elif predicted_class == 1:
        return f"😐 One pair — a common hand. Play cautiously."
    else:
        return f"🃏 Weak hand (nothing). Consider folding unless bluffing."

# Streamlit UI
st.title("♠️ Poker Hand Classifier")
st.write("Select up to **5 cards**. You may leave some as **'Unknown'** if they haven’t been revealed.")

card_inputs = []
for i in range(1, 6):
    col1, col2 = st.columns(2)
    with col1:
        suit = st.selectbox(f"Card {i} Suit", suits, key=f"suit_{i}")
    with col2:
        rank = st.selectbox(f"Card {i} Rank", ranks, key=f"rank_{i}")
    card_inputs.append((suit, rank))

if st.button("Classify Hand"):
    X_input = encode_hand(card_inputs)
    prediction = model.predict(X_input)[0]
    predicted_class = np.argmax(prediction)
    st.subheader(f"🃏 Predicted Hand: **{hand_labels[predicted_class]}**")
    st.info(generate_advice(predicted_class, card_inputs))

    # Confidence scores
    st.markdown("### 🔍 Confidence Scores")
    for i, score in enumerate(prediction):
        st.write(f"{hand_labels[i]}: {score:.3f}")
