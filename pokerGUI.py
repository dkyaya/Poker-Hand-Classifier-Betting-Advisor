import streamlit as st
import numpy as np
import pandas as pd
import random
import time
from collections import Counter
from tensorflow.keras.models import load_model
from openai import OpenAI

# OpenAI client
client = OpenAI(api_key=st.secrets["openai_api_key"])

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

# Suit-specific styling for colors
suit_colors = {"♥": "red", "♦": "red", "♠": "default", "♣": "default"}

def get_betting_advice(hand_types):
    if isinstance(hand_types, list):
        prompt = f"You’re a poker expert. My hand likely falls into one of the following categories: {', '.join(hand_types)}. Give betting advice based on this uncertainty. Limit response to 4 bullet points."
    else:
        prompt = f"You’re a poker expert. I have a hand: {hand_types}. Give clear betting advice. Limit response to 4 bullet points."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a concise poker expert giving betting advice."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# ---- Streamlit UI ----
st.title("🃏 Poker Hand Classifier & Betting Advisor")
st.markdown("Click to select up to 5 cards. You can remove them as needed.")

suits = ["♥", "♠", "♦", "♣"]
rank_labels = ["A"] + [str(n) for n in range(2, 11)] + ["J", "Q", "K"]

if "selected_cards" not in st.session_state:
    st.session_state.selected_cards = []

st.subheader("Card Selector")
selected_set = set(st.session_state.selected_cards)
for s_idx, suit in enumerate(suits):
    cols = st.columns(13)
    for r_idx, rank in enumerate(rank_labels):
        card = (s_idx, r_idx + 1)
        if card in selected_set:
            continue  # Skip already selected cards
        label = f"{rank}{suit}"
        if suit_colors[suit] == "red":
            display_label = f":red[{label}]"
        else:
            display_label = label
        if cols[r_idx].button(display_label):
            if len(st.session_state.selected_cards) < 5:
                st.session_state.selected_cards.append(card)
                st.rerun()

# Show current hand
st.markdown("### 🃏 Selected Cards:")
if st.session_state.selected_cards:
    cols = st.columns(len(st.session_state.selected_cards))
    for i, card in enumerate(st.session_state.selected_cards):
        s, r = card
        label = f"{rank_labels[r-1]}{suits[s]}"
        color = suit_colors[suits[s]]
        if color == "red":
            button_label = f"❌ :red[{label}]"
        else:
            button_label = f"❌ {label}"
        if cols[i].button(button_label):
            st.session_state.selected_cards.remove(card)
            st.rerun()

    if st.button("🧹 Clear Hand"):
        st.session_state.selected_cards.clear()
        st.rerun()
else:
    st.info("Select up to 5 cards above.")

if st.button("Classify Hand"):
    known_cards = st.session_state.selected_cards
    unknown_count = 5 - len(known_cards)

    if unknown_count == 0:
        flat_hand = [val for card in known_cards for val in card]
        input_array = np.array([flat_hand])
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
        spinner = st.empty()
        for i in range(6):
            spinner.info(f"♠ ♦ ♣ ♥ Thinking{'.' * (i % 4)}")
            time.sleep(0.15)

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

        spinner.empty()
        counter = Counter(predictions)
        top_two = counter.most_common(2)
        hand_types = [class_labels[cls] for cls, _ in top_two]

        st.markdown(f"### 🏆 Top Predictions: :orange[{hand_types[0]}] and :orange[{hand_types[1]}]")

        st.markdown("### 🔢 Simulation Breakdown:")
        total = sum(counter.values())
        for cls, count in counter.most_common():
            percent = (count / total) * 100
            st.write(f"- {class_labels[cls]}: {percent:.2f}%")

        st.caption(f"🔢 Total: {sum((count / total) * 100 for count in counter.values()):.2f}%")

        with st.spinner("Asking a poker expert for betting advice..."):
            advice = get_betting_advice(hand_types)
        st.info(f"💡 **AI Betting Advice:** {advice}")
    else:
        st.error("Please select at least one known card to simulate the rest.")
