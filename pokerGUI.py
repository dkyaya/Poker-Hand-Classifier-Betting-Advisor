import streamlit as st
import numpy as np
import pandas as pd
import random
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from openai import OpenAI

client = OpenAI(api_key="sk-proj...")

# OpenAI API key setup

# Load and preprocess data
@st.cache_resource
def load_model_and_encoder():
    model = load_model("pokerhandV7_syn.keras")

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

    return model, one_hot_encode_cards

model, one_hot_encode_cards = load_model_and_encoder()

class_labels = {
    0: "Nothing in hand",
    1: "One pair",
    2: "Two pairs",
    3: "Three of a kind",
    4: "Straight",
    5: "Flush",
    6: "Full house",
    7: "Four of a kind",
    8: "Straight flush",
    9: "Royal flush"
}

def get_betting_advice(hand_types):
    if isinstance(hand_types, list):
        prompt = f"You’re a poker expert. My hand likely falls into one of the following categories: {', '.join(hand_types)}. Give betting advice based on this uncertainty. Limit response to 4 bullet points."
    else:
        prompt = f"You’re a poker expert. I have a hand: {hand_types}. Give clear betting advice. Limit response to 4 bullet points."

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a concise poker expert giving betting advice."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=300,
    temperature=0.7)

    return response.choices[0].message.content.strip()

# ---- Streamlit UI ----
st.title("🃏 Poker Hand Classifier & Betting Advisor")
st.markdown("Input a poker hand one card at a time. You can mark a card as unknown.")

suits = ["Hearts", "Spades", "Diamonds", "Clubs"]
hand_input = []

cols = st.columns(5)
for i, col in enumerate(cols):
    suit = col.selectbox(f"Card {i+1} Suit", options=["Unknown"] + suits, index=0)
    rank = col.selectbox(f"Card {i+1} Rank", options=["Unknown"] + list(range(1, 14)), index=0)

    if suit == "Unknown" or rank == "Unknown":
        hand_input.extend([None, None])
    else:
        hand_input.extend([suits.index(suit), int(rank)])

if st.button("Classify Hand"):
    known_cards = [(hand_input[i], hand_input[i+1]) for i in range(0, 10, 2) if hand_input[i] is not None and hand_input[i+1] is not None]
    unknown_count = 5 - len(known_cards)

    if unknown_count == 0:
        input_array = np.array([hand_input])
        encoded_input = one_hot_encode_cards(input_array)
        prediction = model.predict(encoded_input, verbose=0)
        predicted_class = int(np.argmax(prediction))
        hand_type = class_labels[predicted_class]

        st.success(f"🃏 Predicted Hand Type: **{hand_type}**")

        with st.spinner("Asking a poker expert for betting advice..."):
            advice = get_betting_advice(hand_type)
        st.info(f"💡 **AI Betting Advice:** {advice}")
    elif unknown_count < 5:
        st.warning(f"Only {5 - unknown_count} cards provided. Simulating {unknown_count} unknown cards...")
        all_possible_cards = [(s, r) for s in range(4) for r in range(1, 14)]
        used_cards = set(known_cards)

        predictions = []
        for _ in range(100):
            sampled_hand = known_cards.copy()
            remaining = random.sample([c for c in all_possible_cards if c not in used_cards], unknown_count)
            full_hand = sampled_hand + remaining
            flat_hand = [val for card in full_hand for val in card]
            input_array = np.array([flat_hand])
            encoded_input = one_hot_encode_cards(input_array)
            prediction = model.predict(encoded_input, verbose=0)
            predicted_class = int(np.argmax(prediction))
            predictions.append(predicted_class)

        counter = Counter(predictions)
        top_two = counter.most_common(2)
        hand_types = [class_labels[cls] for cls, _ in top_two]

        st.success(f"🃏 Top Predictions: {hand_types[0]} and {hand_types[1]}")

        st.markdown("### 🔢 Simulation Breakdown:")
        total = sum(counter.values())
        for cls, count in counter.most_common():
            percent = (count / total) * 100
            st.write(f"- {class_labels[cls]}: {percent:.2f}%")

        total_percent = sum((count / total) * 100 for count in counter.values())
        st.caption(f"🔢 Total: {total_percent:.2f}%")

        with st.spinner("Asking a poker expert for betting advice..."):
            advice = get_betting_advice(hand_types)
        st.info(f"💡 **AI Betting Advice:** {advice}")
    else:
        st.error("Please provide at least one known card to simulate the rest.")

def test_model_on_known_hands(model, one_hot_encode_fn, class_labels):
    import numpy as np

    test_hands = {
        0: [0, 2, 1, 5, 2, 8, 3, 11, 0, 13],
        1: [0, 2, 1, 2, 2, 5, 3, 7, 0, 9],
        2: [0, 4, 1, 4, 2, 9, 3, 9, 0, 11],
        3: [0, 6, 1, 6, 2, 6, 3, 10, 0, 11],
        4: [0, 3, 1, 4, 2, 5, 3, 6, 0, 7],
        5: [0, 2, 0, 5, 0, 9, 0, 11, 0, 13],
        6: [0, 7, 1, 7, 2, 7, 0, 10, 1, 10],
        7: [0, 9, 1, 9, 2, 9, 3, 9, 0, 2],
        8: [1, 5, 1, 6, 1, 7, 1, 8, 1, 9],
        9: [0,10, 0,11, 0,12, 0,13, 0, 1]
    }

    correct = 0

    print("\n🧪 Model Evaluation on Known Hands:")
    for true_class, hand in test_hands.items():
        hand_array = np.array([hand])
        encoded = one_hot_encode_fn(hand_array)
        prediction = model.predict(encoded, verbose=0)
        predicted_class = np.argmax(prediction)

        hand_name = class_labels[true_class]
        predicted_name = class_labels[predicted_class]
        status = "✅" if predicted_class == true_class else "❌"
        if status == "✅":
            correct += 1

        print(f"{status} Expected: {true_class} ({hand_name}), Predicted: {predicted_class} ({predicted_name})")

    print(f"\n🏁 Accuracy on known test hands: {correct}/10 ({correct * 10:.1f}%)")

test_model_on_known_hands(model, one_hot_encode_cards, class_labels)
