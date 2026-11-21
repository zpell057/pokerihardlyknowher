"""
Bot template. Copy this folder to my_bots/<your_team>/ and write decide().

  - decide() is called once per decision and has 2 seconds to return, or your
    bot folds. Crashes and invalid actions also fold; your bot stays in the game.
  - You may use the Python 3.10 standard library, eval7, numpy, scipy, treys
    and scikit-learn. Nothing else is installed in the sandbox.
  - No network, no file writes, no subprocesses, no eval/exec. You may read
    your own data/ folder, at import time only.
  - Check your bot before submitting:  make validate BOT=my_bots/<your_team>

The full list of what decide() receives is in the README ("What your bot
sees"); the most useful keys are summarised below.

RETURN FORMAT:
  {"action": "fold"}
  {"action": "check"}                       # only when can_check is True
  {"action": "call"}
  {"action": "raise", "amount": 1200}       # amount = your TOTAL bet this street
  {"action": "all_in"}

CARDS are strings like "As" (ace of spades) or "Td" (ten of diamonds).
"""


import random  # noqa: F401
import json
from pathlib import Path
from collections import defaultdict
from itertools import accumulate

import eval7


# ============================================================
# CONFIGURATION
# ============================================================

RANKS = "23456789TJQKA"
SUITS = "shdc"

EQUITY_ITERATIONS = 2000

# Small smoothing factor prevents a hand from becoming
# completely impossible due to a zero in the lookup table.
RANGE_SMOOTHING = 0.01

# Load precomputed preflop action frequencies.
DATA_PATH = Path(__file__).with_name("preflop_ranges.json")

with open(DATA_PATH) as f:
    PREFLOP_RANGES = json.load(f)


# ============================================================
# CARD AND HAND UTILITIES
# ============================================================

def get_hand_category(card1, card2):
    rank1, suit1 = card1[0], card1[1]
    rank2, suit2 = card2[0], card2[1]

    if rank1 == rank2:
        return rank1 + rank2

    if RANKS.index(rank1) > RANKS.index(rank2):
        high, low = rank1, rank2
    else:
        high, low = rank2, rank1

    return high + low + ("s" if suit1 == suit2 else "o")


# Generate every exact two-card combination.
DECK_STRINGS = [
    rank + suit
    for rank in RANKS
    for suit in SUITS
]

CARD_OBJECTS = {
    card: eval7.Card(card)
    for card in DECK_STRINGS
}

ALL_COMBINATIONS = []

for i in range(len(DECK_STRINGS)):
    for j in range(i + 1, len(DECK_STRINGS)):
        card1 = DECK_STRINGS[i]
        card2 = DECK_STRINGS[j]

        category = get_hand_category(card1, card2)

        ALL_COMBINATIONS.append(
            (card1, card2, category)
        )


# ============================================================
# OPPONENT RANGE
# ============================================================

def get_first_preflop_action(game_state, player):
    """
    Find the opponent's first meaningful preflop action.

    The lookup table only describes a player acting first,
    so we only use raise/check/fold as informative actions.
    """

    player_id = player.get("bot_id")
    player_seat = player.get("seat")

    for entry in game_state.get("action_log", []):

        if entry.get("street") != "preflop":
            continue

        same_player = (
            player_id is not None
            and entry.get("bot_id") == player_id
        )

        if player_id is None:
            same_player = entry.get("seat") == player_seat

        if not same_player:
            continue

        action = entry.get("action")

        if action in ("raise", "check", "fold", "call", "all_in"):
            return action

    return None


def get_opponent_weights(action):
    """
    Return the likelihood of each exact hand given
    the opponent's observed first preflop action.

    An unknown or unsupported action gets a neutral weight.
    """

    weights = []

    for card1, card2, category in ALL_COMBINATIONS:

        probabilities = PREFLOP_RANGES.get(category, {})

        if action in ("raise", "check", "fold"):
            likelihood = probabilities.get(action, 1.0)
        else:
            likelihood = 1.0

        weight = likelihood + RANGE_SMOOTHING

        weights.append(
            (card1, card2, weight)
        )

    return weights


def build_opponent_ranges(game_state):
    """
    Build one weighted range for every opponent still in the hand.

    Folded players are excluded.
    """

    players = game_state.get("players", [])
    my_id = game_state.get("your_bot_id")
    my_seat = game_state.get("seat_to_act")

    opponent_ranges = []

    for player in players:

        if player.get("bot_id") == my_id:
            continue

        if (
            player.get("bot_id") is None
            and player.get("seat") == my_seat
        ):
            continue

        state = str(player.get("state", "active")).lower()

        if state in ("folded", "out", "sitting_out"):
            continue

        action = get_first_preflop_action(
            game_state,
            player
        )

        weights = get_opponent_weights(action)

        opponent_ranges.append(weights)

    # Fallback if player information is unavailable.
    if not opponent_ranges:
        opponent_ranges.append(
            get_opponent_weights(None)
        )

    return opponent_ranges


# ============================================================
# RANGE-WEIGHTED MONTE CARLO EQUITY
# ============================================================

def calculate_range_equity(
    hero_cards,
    board,
    opponent_ranges,
    iterations=EQUITY_ITERATIONS
):
    """
    Estimate showdown equity against one or more opponents.

    Opponent hands are sampled according to their weighted ranges.
    Cards cannot overlap between players.
    """

    hero = [
        eval7.Card(card)
        for card in hero_cards
    ]

    community = [
        eval7.Card(card)
        for card in board
    ]

    known_strings = set(hero_cards + board)

    available_deck = [
        CARD_OBJECTS[card]
        for card in DECK_STRINGS
        if card not in known_strings
    ]

    missing = 5 - len(community)

    # The hands an opponent can hold only depend on the known cards,
    # so filter each range and total its weights once, not every iteration.
    opponent_samplers = []

    for opponent_range in opponent_ranges:

        legal_hands = [
            (c1, c2, weight)
            for c1, c2, weight in opponent_range
            if c1 not in known_strings
            and c2 not in known_strings
        ]

        if not legal_hands:
            continue

        cum_weights = list(
            accumulate(hand[2] for hand in legal_hands)
        )

        opponent_samplers.append(
            (legal_hands, cum_weights)
        )

    equity_total = 0.0

    for _ in range(iterations):

        used_cards = set(known_strings)

        villain_hands = []

        # Sample one hand for each opponent.
        for legal_hands, cum_weights in opponent_samplers:

            # Redraw if an earlier opponent already holds one of these cards.
            while True:

                card1, card2, _ = random.choices(
                    legal_hands,
                    cum_weights=cum_weights,
                    k=1
                )[0]

                if (
                    card1 not in used_cards
                    and card2 not in used_cards
                ):
                    break

            used_cards.add(card1)
            used_cards.add(card2)

            villain_hands.append([
                CARD_OBJECTS[card1],
                CARD_OBJECTS[card2]
            ])

        # Complete the board.
        remaining_deck = [
            card
            for card in available_deck
            if str(card) not in used_cards
        ]

        runout = random.sample(
            remaining_deck,
            missing
        )

        final_board = community + runout

        hero_score = eval7.evaluate(
            hero + final_board
        )

        villain_scores = [
            eval7.evaluate(villain + final_board)
            for villain in villain_hands
        ]

        if not villain_scores:
            equity_total += 1.0
            continue

        best_villain_score = max(villain_scores)

        if hero_score > best_villain_score:
            equity_total += 1.0

        elif hero_score == best_villain_score:

            tied_villains = villain_scores.count(
                hero_score
            )

            equity_total += 1.0 / (tied_villains + 1)

    return equity_total / iterations


# ============================================================
# EXPECTED VALUE
# ============================================================

def calculate_call_ev(equity, pot, amount_owed):

    final_pot = pot + amount_owed

    return equity * final_pot - amount_owed


# ============================================================
# MAIN DECISION FUNCTION
# ============================================================

def decide(game_state: dict) -> dict:

    if game_state.get("type") == "warmup":
        return {"action": "check"}

    cards = game_state["your_cards"]
    board = game_state["community_cards"]

    pot = game_state["pot"]
    owed = game_state["amount_owed"]

    # Build opponent ranges from observed preflop actions.
    opponent_ranges = build_opponent_ranges(
        game_state
    )

    # Calculate equity against those ranges.
    equity = calculate_range_equity(
        cards,
        board,
        opponent_ranges
    )

    call_ev = calculate_call_ev(
        equity,
        pot,
        owed
    )

    debug = {
        "equity": round(equity, 3),
        "call_ev": round(call_ev, 2),
        "opponents": len(opponent_ranges)
    }

    # Nobody has bet: check or raise.
    if owed == 0:

        if equity > 0.65:

            raise_to = max(
                game_state["min_raise_to"],
                game_state["current_bet"] + int(pot * 0.5)
            )

            if raise_to <= (
                game_state["your_stack"]
                + game_state["your_bet_this_street"]
            ):

                return {
                    "action": "raise",
                    "amount": raise_to,
                    "debug": debug
                }

        return {
            "action": "check",
            "debug": debug
        }

    # Facing a bet: fold if calling has negative EV.
    if call_ev <= 0:

        return {
            "action": "fold",
            "debug": debug
        }

    # Strong hand: consider raising.
    if equity > 0.65:

        raise_to = max(
            game_state["min_raise_to"],
            game_state["current_bet"] + int(pot * 0.5)
        )

        if raise_to <= (
            game_state["your_stack"]
            + game_state["your_bet_this_street"]
        ):

            return {
                "action": "raise",
                "amount": raise_to,
                "debug": debug
            }

    return {
        "action": "call",
        "debug": debug
    }
