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

import math
import random  # noqa: F401  (seeded per match, so a seeded match replays exactly)
import eval7


def calculate_equity(hero_cards, board, iterations=2000):
    hero = [eval7.Card(c) for c in hero_cards]
    community = [eval7.Card(c) for c in board]

    deck = eval7.Deck()

    known = hero + community
    for card in known:
        deck.cards.remove(card)

    wins = 0
    ties = 0

    missing = 5 - len(community)

    for _ in range(iterations):
        remaining = random.sample(deck.cards, 2 + missing)

        villain = remaining[:2]
        runout = remaining[2:]

        final_board = community + runout

        hero_score = eval7.evaluate(hero + final_board)
        villain_score = eval7.evaluate(villain + final_board)

        if hero_score > villain_score:
            wins += 1
        elif hero_score == villain_score:
            ties += 1

    return (wins + ties / 2) / iterations

def calculate_call_ev(equity, pot, amount_owed):
    final_pot = pot + amount_owed

    call_ev = equity * final_pot - amount_owed

    return call_ev


# ============================================================
# OPPONENT TRACKING
# ============================================================

STREETS = ("preflop", "flop", "turn", "river")

# match_action_log only holds the last 200 entries, so statistics are
# accumulated here; module state lasts exactly one match.
MATCH_LOG_MAX_ENTRIES = 200

opponent_stats = {}
last_processed_hand = -1

# Per opponent: showdowns after they bet or raised, and how many of those
# hands were weak (bluffs).
bluff_stats = {}

# The last board this bot saw in each hand, to judge hands shown at showdown.
# The log of a showdown lists the cards shown but not the board.
seen_boards = {}


def new_street_stats():
    return {
        "folds": 0,
        "fold_opportunities": 0,
        "raises": 0,
        "calls": 0
    }


def get_opponent_stats(bot_id):
    if bot_id not in opponent_stats:
        opponent_stats[bot_id] = {
            street: new_street_stats()
            for street in STREETS
        }

    return opponent_stats[bot_id]


def record_hand(entries, my_id, board=None):
    """
    Add the opponents' decisions from one completed hand to opponent_stats,
    and any showdown after they bet or raised to bluff_stats.

    The betting is replayed street by street so each decision knows whether
    the player owed chips to a bet or raise. Blind posts set the amount to
    call but are not pressure on their own; antes are dead money.
    """

    street = None
    current_bet = 0
    bets = {}
    raised = False

    aggressors = set()
    shown = {}

    for entry in entries:

        entry_street = entry.get("street")
        player = entry.get("bot_id")

        if entry.get("action") == "show" and player is not None:
            shown[player] = entry.get("cards") or []
            continue

        # Skips showdown reveals, which use the street "showdown".
        if entry_street not in STREETS or player is None:
            continue

        if entry_street != street:
            street = entry_street
            current_bet = 0
            bets = {}
            raised = False

        action = entry.get("action")
        amount = entry.get("amount") or 0

        if action in ("small_blind", "big_blind"):
            bets[player] = bets.get(player, 0) + amount
            current_bet = max(current_bet, bets[player])
            continue

        if action not in ("fold", "check", "call", "raise", "all_in"):
            continue

        owed = current_bet - bets.get(player, 0)

        # For raise and all_in, amount is the player's total on this street;
        # an all_in that doesn't exceed the current bet is a call.
        is_raise = (
            action in ("raise", "all_in")
            and amount > current_bet
        )

        if player != my_id:

            stats = get_opponent_stats(player)[street]

            if raised and owed > 0:
                stats["fold_opportunities"] += 1

                if action == "fold":
                    stats["folds"] += 1

            if is_raise:
                stats["raises"] += 1
                aggressors.add(player)
            elif action in ("call", "all_in"):
                stats["calls"] += 1

        if action == "call":
            bets[player] = bets.get(player, 0) + amount

        elif action in ("raise", "all_in"):
            bets[player] = max(bets.get(player, 0), amount)

            if is_raise:
                current_bet = amount
                raised = True

    record_bluffs(aggressors, shown, board or [])


def record_bluffs(aggressors, shown, board):
    """
    A player who bet or raised and then showed a hand with little equity
    against a random hand, on the last board we saw, was bluffing.
    """

    for player in aggressors:

        cards = shown.get(player)

        if len(cards or []) != 2:
            continue

        stats = bluff_stats.setdefault(
            player, {"aggressive_showdowns": 0, "bluffs": 0}
        )
        stats["aggressive_showdowns"] += 1

        if calculate_equity(cards, board, BLUFF_EQUITY_ITERATIONS) < BLUFF_EQUITY:
            stats["bluffs"] += 1


def get_bluff_rate(bot_id):
    """Share of this opponent's aggressive showdowns that were bluffs, or
    None until there are MIN_BLUFF_OBSERVATIONS of them."""

    stats = bluff_stats.get(bot_id)

    if not stats or stats["aggressive_showdowns"] < MIN_BLUFF_OBSERVATIONS:
        return None

    return stats["bluffs"] / stats["aggressive_showdowns"]


def update_opponent_stats(game_state):
    """
    Record every completed hand in match_action_log that hasn't been
    recorded yet. The current hand is still in progress, so it waits
    until a later decision.
    """

    global last_processed_hand

    log = game_state.get("match_action_log") or []
    current_hand = game_state.get("hand_num", 0)
    my_id = game_state.get("your_bot_id")

    for player in game_state.get("players", []):
        bot_id = player.get("bot_id")

        if bot_id is not None and bot_id != my_id:
            get_opponent_stats(bot_id)

    hands = {}

    for entry in log:
        hand = entry.get("hand_num")

        if hand is None:
            continue

        if last_processed_hand < hand < current_hand:
            hands.setdefault(hand, []).append(entry)

    # A full log may have cut the oldest hand off partway through.
    if log and len(log) >= MATCH_LOG_MAX_ENTRIES:
        hands.pop(log[0].get("hand_num"), None)

    for hand in sorted(hands):
        record_hand(hands[hand], my_id, seen_boards.get(hand))

    last_processed_hand = max(last_processed_hand, current_hand - 1)

    for hand in [h for h in seen_boards if h <= last_processed_hand]:
        del seen_boards[hand]


def summarize_opponent_stats():
    """Per-opponent, per-street rates and sample sizes for debug output."""

    summary = {}

    for bot_id, streets in opponent_stats.items():

        summary[bot_id] = {}

        for street in STREETS:

            stats = streets[street]
            observed = stats["raises"] + stats["calls"]

            summary[bot_id][street] = {
                # Laplace smoothing: 0.5 with no observations.
                "fold_rate": round(
                    (stats["folds"] + 1)
                    / (stats["fold_opportunities"] + 2),
                    3
                ),
                "folds": stats["folds"],
                "fold_n": stats["fold_opportunities"],
                "aggression_rate": (
                    round(stats["raises"] / observed, 3)
                    if observed else None
                ),
                "raises": stats["raises"],
                "aggression_n": observed
            }

    return summary


# ============================================================
# OPPONENT-ADJUSTED THRESHOLDS
# ============================================================

BASE_RAISE_THRESHOLD = 0.65

# A rate must rest on this many observations before it moves a
# threshold; the smoothed fold rate of a single fold is already 0.67.
MIN_OBSERVATIONS = 5


def get_heads_up_opponent(game_state):
    """The one opponent still in the hand, or None when there are several."""

    my_id = game_state.get("your_bot_id")

    opponents = [
        player.get("bot_id")
        for player in game_state.get("players", [])
        if player.get("bot_id") != my_id
        and player.get("state") in ("active", "all_in")
    ]

    return opponents[0] if len(opponents) == 1 else None


def get_raise_threshold(stats, street, facing_bet):
    """
    Equity needed to raise on this street, adjusted for the opponent's
    history. stats is one opponent's entry from summarize_opponent_stats().
    """

    threshold = BASE_RAISE_THRESHOLD

    if not stats or street not in stats:
        return threshold

    street_stats = stats[street]

    if not facing_bet:
        # Opponents who often fold to bets are good targets: bet thinner.
        if street_stats["fold_n"] >= MIN_OBSERVATIONS:
            fold_rate = street_stats["fold_rate"]

            if fold_rate > 0.65:
                threshold -= 0.10
            elif fold_rate > 0.50:
                threshold -= 0.05

    else:
        # Against frequent bettors, calling keeps their bluffs in the pot;
        # against passive players, a bet usually means a strong hand.
        # Either way, raise only with more equity.
        if street_stats["aggression_n"] >= MIN_OBSERVATIONS:
            aggression_rate = street_stats["aggression_rate"]

            if aggression_rate > 0.60:
                threshold += 0.05
            elif aggression_rate < 0.20:
                threshold += 0.05

    return threshold


# ============================================================
# OPPONENT CONFIDENCE
# ============================================================

# Equity against a random hand overstates our chances once the opponent
# has shown strength. effective equity = equity * (1 - ALPHA * confidence).
CONFIDENCE_ALPHA = 0.3

# Bets on later streets are more often honest.
STREET_WEIGHTS = {"preflop": 0.6, "flop": 1.0, "turn": 1.3, "river": 1.6}

CALL_WEIGHT = 0.5
CHECK_WEIGHT = -0.3

# Rate assumed until a street has MIN_OBSERVATIONS for this opponent.
UNKNOWN_RATE = 0.5

# A shown hand below this equity against a random hand, after betting, was
# a bluff. Bluff rates count only after this many such showdowns.
BLUFF_EQUITY = 0.40
BLUFF_EQUITY_ITERATIONS = 400
MIN_BLUFF_OBSERVATIONS = 3


def known_rate(street_stats, rate_key, count_key):
    """A rate from summarize_opponent_stats(), or UNKNOWN_RATE while it rests
    on too few observations."""

    if not street_stats or street_stats[count_key] < MIN_OBSERVATIONS:
        return UNKNOWN_RATE

    rate = street_stats[rate_key]

    return UNKNOWN_RATE if rate is None else rate


def estimate_opponent_confidence(game_state, opponent_id, stats):
    """
    How strong the opponent's actions this hand have been, from 0 to 1.

    Each action adds street_weight * action_weight * surprise:
      raise/bet  1 + min(bet / pot, 1.5), surprise 1 - their aggression rate
      call       CALL_WEIGHT,             surprise their fold rate
      check      CHECK_WEIGHT,            surprise 1
    and the total is squashed with 1 - exp(-total). A bluff rate seen at
    showdown discounts the result.

    stats is the opponent's entry from summarize_opponent_stats().
    """

    if opponent_id is None:
        return 0.0

    signal = 0.0
    pot = 0
    street = None
    current_bet = 0
    bets = {}

    for entry in game_state.get("action_log", []):

        entry_street = entry.get("street")
        player = entry.get("bot_id")
        action = entry.get("action")
        amount = entry.get("amount") or 0

        if entry_street not in STREETS:
            continue

        if entry_street != street:
            street = entry_street
            current_bet = 0
            bets = {}

        # Chips this action puts in. For raise and all_in, amount is the
        # player's total on the street; for posts and calls, the chips added.
        if action in ("ante", "small_blind", "big_blind", "call"):
            added = amount
        elif action in ("raise", "all_in"):
            added = max(amount - bets.get(player, 0), 0)
        else:
            added = 0

        is_raise = action in ("raise", "all_in") and amount > current_bet

        if player == opponent_id and action in ("check", "call", "raise", "all_in"):

            street_stats = (stats or {}).get(street)
            weight = STREET_WEIGHTS[street]

            if is_raise:
                size = min(added / pot, 1.5) if pot > 0 else 1.0
                aggression = known_rate(street_stats, "aggression_rate", "aggression_n")
                signal += weight * (1.0 + size) * (1.0 - aggression)

            elif action in ("call", "all_in"):
                fold_rate = known_rate(street_stats, "fold_rate", "fold_n")
                signal += weight * CALL_WEIGHT * fold_rate

            elif action == "check":
                signal += weight * CHECK_WEIGHT

        pot += added

        if action != "ante":
            bets[player] = bets.get(player, 0) + added
            current_bet = max(current_bet, bets[player])

    confidence = 1.0 - math.exp(-max(signal, 0.0))

    bluff_rate = get_bluff_rate(opponent_id)

    if bluff_rate is not None:
        confidence *= 1.0 - bluff_rate

    return confidence


def decide(game_state: dict) -> dict:
    """
    game_state (the most useful keys):
      your_cards        ["Ah", "Kh"]
      community_cards   [] preflop, then 3, 4, 5 cards
      street            "preflop" | "flop" | "turn" | "river"
      pot, current_bet, min_raise_to, amount_owed, can_check
      your_stack, your_bet_this_street, your_bot_id
      blinds            {"sb", "bb", "ante"} for this hand (they rise every 20 hands)
      hand_num          hand number in this match, from 0
      button_seat, sb_seat, bb_seat, seat_to_act
      players           public info per seat: seat, bot_id, stack, state, bet_this_street, ...
      action_log        every action this hand: {"seat", "bot_id", "action", "amount", "street"}
      match_action_log  the last 200 actions across the match, with "hand_num", plus
                        {"action": "show", "cards": [...]} for each player at a showdown

    Debugging: anything you print() is saved with the decision, and you can add
    a JSON-serialisable "debug" value to what you return, e.g.
    {"action": "call", "debug": {"equity": 0.4}}. Run with: python3 play.py --show
    """

    if game_state.get("type") == "warmup":
        return {"action": "check"}

    cards = game_state["your_cards"]
    board = game_state["community_cards"]

    pot = game_state["pot"]
    owed = game_state["amount_owed"]

    seen_boards[game_state.get("hand_num", 0)] = list(board)

    update_opponent_stats(game_state)

    opponent_summary = summarize_opponent_stats()
    opponent_id = get_heads_up_opponent(game_state)
    opponent = opponent_summary.get(opponent_id)

    raw_equity = calculate_equity(cards, board)

    confidence = estimate_opponent_confidence(game_state, opponent_id, opponent)

    equity = raw_equity * (1.0 - CONFIDENCE_ALPHA * confidence)

    call_ev = calculate_call_ev(equity, pot, owed)

    raise_threshold = get_raise_threshold(
        opponent,
        game_state["street"],
        facing_bet=owed > 0
    )

    debug = {
        "raw_equity": round(raw_equity, 3),
        "confidence": round(confidence, 3),
        "equity": round(equity, 3),
        "call_ev": round(call_ev, 2),
        "raise_threshold": round(raise_threshold, 3),
        "bluff_rate": get_bluff_rate(opponent_id),
        "opponent_stats": opponent_summary
    }

    # Nobody has bet: check or raise
    if owed == 0:

        if equity > raise_threshold:
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

    # Facing a bet: fold or call based on EV
    if call_ev <= 0:
        return {
            "action": "fold",
            "debug": debug
        }

    # Strong hand: consider raising
    if equity > raise_threshold:
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

