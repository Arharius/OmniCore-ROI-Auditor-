from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import networkx as nx
from scipy import stats
import pandas as pd


@dataclass
class GraphResult:
    bottleneck_node: str
    bottleneck_score: float
    pagerank: dict
    betweenness: dict
    all_nodes_ranked: list


# ── Markov Graph (data-driven ETL) ────────────────────────────────────────────

@dataclass
class ReworkPair:
    """A detected bidirectional (loop/rework) transition between two stages."""
    stage_a: str
    stage_b: str
    prob_a_to_b: float      # P(transition A → B)
    prob_b_to_a: float      # P(rework B → A | task is currently in B)
    avg_days_in_b: float    # average time_spent in B before going back to A
    count_rework: int       # number of rework events (B → A occurrences)


@dataclass
class MarkovGraphResult:
    """Full result of the Markov-chain graph analysis from a normalised DataFrame."""
    # ── Transition data ───────────────────────────────────────────────────────
    transition_probs:  Dict[Tuple[str, str], float]     # (from, to) → probability
    transition_counts: Dict[Tuple[str, str], int]       # (from, to) → raw count
    stage_total:       Dict[str, int]                   # stage → total outgoing

    # ── Rework / Loop analysis ────────────────────────────────────────────────
    rework_pairs:    List[ReworkPair]       # sorted by avg_days_in_b desc
    rework_edges:    Set[Tuple[str, str]]  # set of (from, to) that are rework
    rework_rate:     Dict[str, float]      # stage → fraction of outgoing that loop
    decision_latency: Dict[str, float]     # stage → avg days wasted before rework

    # ── Bottleneck ────────────────────────────────────────────────────────────
    bottleneck_node:         str
    bottleneck_rework_rate:  float    # rework rate AT the bottleneck stage
    bottleneck_avg_days:     float    # avg days spent in bottleneck stage
    bottleneck_score:        float    # composite score used for ranking

    # ── NetworkX graph ────────────────────────────────────────────────────────
    G:            Any                   # nx.DiGraph
    betweenness:  Dict[str, float]
    pagerank:     Dict[str, float]
    combo_score:  Dict[str, float]      # composite score per node

    # ── Summary counts ────────────────────────────────────────────────────────
    total_rework_transitions: int
    total_transitions:        int


@dataclass
class MarkovResult:
    expected_lead_time_hours: float
    p_complete: float
    p_lost: float
    fundamental_matrix: np.ndarray
    states: list


@dataclass
class BayesResult:
    prior_pct: float
    posterior_pct: float
    ci_80_low: float
    ci_80_high: float


def build_markov_graph(df: pd.DataFrame) -> MarkovGraphResult:
    """
    Build a Markov-chain transition graph from a normalised process DataFrame.

    Expected columns: entity_id | current_stage | next_stage | time_spent (days)

    Returns a MarkovGraphResult with:
      - Transition probability matrix
      - Rework / loop detection with Decision Latency per pair
      - Bottleneck node (composite score: betweenness + rework rate)
      - NetworkX DiGraph with is_rework edge attribute
    """
    if df is None or len(df) == 0:
        raise ValueError("Empty DataFrame passed to build_markov_graph")

    # ── 0. Drop rows with NaN / empty stage names ───────────────────────────
    df = df.dropna(subset=["current_stage", "next_stage"])
    df = df[df["current_stage"].astype(str).str.strip() != ""]
    df = df[df["next_stage"].astype(str).str.strip() != ""]
    df = df[df["current_stage"].astype(str).str.lower() != "nan"]
    df = df[df["next_stage"].astype(str).str.lower() != "nan"]
    if len(df) == 0:
        raise ValueError("No valid rows after dropping NaN/empty stage values")

    # ── 1. Raw transition counts ────────────────────────────────────────────
    tc = (
        df.groupby(["current_stage", "next_stage"])
        .agg(count=("time_spent", "size"), avg_time=("time_spent", "mean"))
        .reset_index()
    )
    stage_total: Dict[str, int] = df.groupby("current_stage").size().to_dict()

    # Build dicts keyed on (from, to) tuples
    transition_counts: Dict[Tuple[str, str], int] = {}
    transition_probs:  Dict[Tuple[str, str], float] = {}
    for _, row in tc.iterrows():
        frm, to, cnt = str(row["current_stage"]), str(row["next_stage"]), int(row["count"])
        transition_counts[(frm, to)] = cnt
        total = stage_total.get(frm, 1)
        transition_probs[(frm, to)] = round(cnt / total, 4)

    existing: Set[Tuple[str, str]] = set(transition_counts.keys())

    # ── 2. Detect rework edges: (A→B) is rework if (B→A) also exists ──────
    rework_edges: Set[Tuple[str, str]] = {
        (frm, to) for frm, to in existing if (to, frm) in existing
    }

    # ── 3. Build ReworkPair records ─────────────────────────────────────────
    seen: Set[frozenset] = set()
    rework_pairs: List[ReworkPair] = []
    for frm, to in rework_edges:
        pair_key = frozenset({frm, to})
        if pair_key in seen:
            continue
        seen.add(pair_key)

        # rows where the task spent time in `to` and then went back to `frm`
        rework_mask = (df["current_stage"] == to) & (df["next_stage"] == frm)
        avg_days = float(df[rework_mask]["time_spent"].mean()) if rework_mask.any() else 0.0
        count_rework = int(rework_mask.sum())

        rework_pairs.append(ReworkPair(
            stage_a=frm,
            stage_b=to,
            prob_a_to_b=transition_probs.get((frm, to), 0.0),
            prob_b_to_a=transition_probs.get((to, frm), 0.0),
            avg_days_in_b=round(avg_days, 2),
            count_rework=count_rework,
        ))

    rework_pairs.sort(key=lambda r: r.avg_days_in_b, reverse=True)

    # ── 4. Per-stage rework rate and decision latency ───────────────────────
    rework_rate: Dict[str, float] = {}
    decision_latency: Dict[str, float] = {}

    all_stages = (
        {s for s in df["current_stage"].unique() if isinstance(s, str) and s.strip()}
        | {s for s in df["next_stage"].unique() if isinstance(s, str) and s.strip()}
    )
    for stage in all_stages:
        out_edges = [(frm, to) for frm, to in existing if frm == stage]
        rework_out = [(frm, to) for frm, to in out_edges if (frm, to) in rework_edges]
        rework_rate[stage] = len(rework_out) / len(out_edges) if out_edges else 0.0

        # decision latency: avg time spent in this stage before any rework departure
        rw_mask = (df["current_stage"] == stage) & (
            df["next_stage"].isin([to for _, to in rework_out])
        )
        decision_latency[stage] = (
            float(df[rw_mask]["time_spent"].mean()) if rw_mask.any() else 0.0
        )

    # ── 5. Build NetworkX DiGraph ────────────────────────────────────────────
    G = nx.DiGraph()
    G.add_nodes_from(all_stages)
    for (frm, to), prob in transition_probs.items():
        G.add_edge(
            frm, to,
            weight=prob,
            count=transition_counts[(frm, to)],
            is_rework=((frm, to) in rework_edges),
        )

    # ── 6. Centrality metrics ────────────────────────────────────────────────
    betweenness = nx.betweenness_centrality(G, weight="weight")
    try:
        pagerank = nx.pagerank(G, weight="weight", max_iter=200)
    except Exception:
        pagerank = {n: 1.0 / max(len(G.nodes), 1) for n in G.nodes}

    # ── 7. Composite bottleneck score: betweenness + rework_rate ────────────
    max_bet = max(betweenness.values()) if betweenness else 1.0
    combo_score: Dict[str, float] = {}
    for node in G.nodes:
        norm_bet = betweenness[node] / max_bet if max_bet > 0 else 0.0
        combo_score[node] = round(0.55 * norm_bet + 0.45 * rework_rate.get(node, 0.0), 5)

    bottleneck_node = max(combo_score, key=combo_score.get) if combo_score else ""

    # ── 8. Bottleneck statistics ─────────────────────────────────────────────
    bt_mask = df["current_stage"] == bottleneck_node
    bottleneck_avg_days = (
        float(df[bt_mask]["time_spent"].mean()) if bt_mask.any() else 0.0
    )
    bottleneck_rework_rate = rework_rate.get(bottleneck_node, 0.0)
    bottleneck_score = combo_score.get(bottleneck_node, 0.0)

    # ── 9. Total rework transition count ────────────────────────────────────
    rework_pair_set = {(rp.stage_b, rp.stage_a) for rp in rework_pairs} | \
                      {(rp.stage_a, rp.stage_b) for rp in rework_pairs}
    total_rework = int(
        df.apply(
            lambda r: (str(r["current_stage"]), str(r["next_stage"])) in rework_edges,
            axis=1,
        ).sum()
    )

    return MarkovGraphResult(
        transition_probs=transition_probs,
        transition_counts=transition_counts,
        stage_total=stage_total,
        rework_pairs=rework_pairs,
        rework_edges=rework_edges,
        rework_rate=rework_rate,
        decision_latency=decision_latency,
        bottleneck_node=bottleneck_node,
        bottleneck_rework_rate=round(bottleneck_rework_rate, 4),
        bottleneck_avg_days=round(bottleneck_avg_days, 2),
        bottleneck_score=round(bottleneck_score, 5),
        G=G,
        betweenness=betweenness,
        pagerank=pagerank,
        combo_score=combo_score,
        total_rework_transitions=total_rework,
        total_transitions=len(df),
    )


class MathEngine:
    """Математический движок для анализа данных ROI."""

    def graph_bottleneck(self, edges: list) -> object:
        """
        Анализирует направленный граф для выявления узких мест.

        Параметры:
            edges: список кортежей (from_node, to_node, weight)

        Возвращает:
            GraphResult с узлом-узким местом, оценками центральности и рейтингом.
        """
        G = nx.DiGraph()
        for from_node, to_node, weight in edges:
            G.add_edge(from_node, to_node, weight=weight)

        betweenness = nx.betweenness_centrality(G, weight="weight")
        pagerank = nx.pagerank(G, weight="weight")

        bottleneck_node = max(betweenness, key=betweenness.get)
        bottleneck_score = round(betweenness[bottleneck_node], 4)

        all_nodes_ranked = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

        return GraphResult(
            bottleneck_node=bottleneck_node,
            bottleneck_score=bottleneck_score,
            pagerank=pagerank,
            betweenness=betweenness,
            all_nodes_ranked=all_nodes_ranked,
        )

    def compute_friction_tax(
        self,
        decision_latency_days: float,
        cost_per_hour: float,
        hours_per_day: float,
    ) -> float:
        """
        Вычисляет «налог на трение» — прямые потери на одну сущность из-за
        задержек решений в петлях доработки.

        Формула:  friction_tax = decision_latency_days × hours_per_day × cost_per_hour

        Параметры:
            decision_latency_days: среднее кол-во дней в узловой стадии перед возвратом
            cost_per_hour:         стоимость одного часа работы команды ($)
            hours_per_day:         рабочих часов в сутках

        Возвращает:
            float: потери в долларах на одну сущность (сделку / задачу)
        """
        if decision_latency_days <= 0 or cost_per_hour <= 0 or hours_per_day <= 0:
            return 0.0
        return round(decision_latency_days * hours_per_day * cost_per_hour, 2)

    def compute_process_confidence(
        self,
        prior_pct: float,
        bottleneck_rework_rate: float,
    ) -> dict:
        """
        Байесовское обновление уверенности в успехе процесса с учётом
        наблюдаемой ставки доработки на узком месте.

        Модель:
          • Evidence   = наблюдаемая ставка доработки (bottleneck_rework_rate)
          • P(ev|success) = 1 − rework_rate  (успешный процесс → мало возвратов)
          • P(ev|failure) = rework_rate       (неуспешный процесс → много возвратов)
          • Likelihood ratio = P(ev|success) / P(ev|failure)
          • Posterior (odds form): prior_odds × LR → posterior_pct

        Параметры:
            prior_pct:              исходная уверенность пользователя, % (0–100)
            bottleneck_rework_rate: доля rework-переходов на узком месте (0–1)

        Возвращает:
            dict с ключами: prior_pct, likelihood_ratio, posterior_pct, delta_pct
        """
        prior = max(min(prior_pct / 100.0, 0.9999), 0.0001)
        rw    = max(min(bottleneck_rework_rate, 0.9999), 0.0001)

        p_ev_success = 1.0 - rw
        p_ev_failure = rw
        likelihood_ratio = p_ev_success / p_ev_failure

        prior_odds     = prior / (1.0 - prior)
        posterior_odds = prior_odds * likelihood_ratio
        posterior      = posterior_odds / (1.0 + posterior_odds)
        posterior_pct  = round(posterior * 100.0, 1)

        return {
            "prior_pct":        round(prior_pct, 1),
            "likelihood_ratio": round(likelihood_ratio, 4),
            "posterior_pct":    posterior_pct,
            "delta_pct":        round(posterior_pct - prior_pct, 1),
        }

    def markov_absorbing(
        self,
        Q: np.ndarray,
        state_times: np.ndarray,
        states: list,
        p_complete_before: float = 0.74,
        p_complete_after: float = 0.96,
    ) -> object:
        """
        Вычисляет ожидаемое время выполнения процесса через поглощающую цепь Маркова.

        Параметры:
            Q: матрица переходных вероятностей между непоглощающими состояниями
            state_times: вектор времени пребывания в каждом состоянии (в часах)
            states: список названий состояний
            p_complete_before: вероятность завершения до (по умолчанию 0.74)
            p_complete_after: вероятность завершения после (по умолчанию 0.96)

        Возвращает:
            MarkovResult с ожидаемым временем выполнения и вероятностями.

        Исключения:
            ValueError: если матрица вырождена
        """
        I = np.eye(Q.shape[0])
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            raise ValueError("Матрица вырождена: невозможно вычислить обратную матрицу (I - Q).")

        if np.linalg.matrix_rank(I - Q) < Q.shape[0]:
            raise ValueError("Матрица вырождена: (I - Q) необратима.")

        expected_lead_time = float(np.sum(N @ state_times))

        return MarkovResult(
            expected_lead_time_hours=expected_lead_time,
            p_complete=p_complete_after,
            p_lost=round(1.0 - p_complete_after, 4),
            fundamental_matrix=N,
            states=states,
        )

    def bayesian_update(
        self,
        positive_signals: int,
        total_signals: int,
        prior_rate: float = 0.34,
    ) -> object:
        """
        Байесовское обновление оценки вероятности успеха.
        Модель: Beta-Bernoulli с эффективным размером априорной выборки n_prior=10.

        Параметры:
            positive_signals: количество положительных сигналов
            total_signals:    общее количество сигналов
            prior_rate:       априорная вероятность µ₀ ∈ (0, 1), по умолчанию 0.34

        Формулы:
            α₀ = µ₀ · n_prior,   β₀ = (1 − µ₀) · n_prior
            αₙ = α₀ + positives, βₙ = β₀ + (total − positives)
            µₙ = αₙ / (αₙ + βₙ)   ← среднее Beta-распределения (аналитически)

        Гарантия направления:
            evidence_rate > µ₀  →  µₙ > µ₀  (апостериор не может упасть ниже приора,
            если доказательство сильнее приора)

        Возвращает:
            BayesResult с prior_pct, posterior_pct и 80% CI.
        """
        prior_mu = max(min(float(prior_rate), 0.9999), 0.0001)
        n_prior  = 10

        alpha_prior = prior_mu * n_prior
        beta_prior  = (1.0 - prior_mu) * n_prior

        alpha_post  = alpha_prior + positive_signals
        beta_post   = beta_prior  + (total_signals - positive_signals)

        # ── Среднее Beta-распределения: E[Beta(α,β)] = α / (α + β) ────────
        prior_mu_calc = alpha_prior / (alpha_prior + beta_prior)
        posterior_mu  = alpha_post  / (alpha_post  + beta_post)

        # ── Сэнити-чек: направление апостериора должно совпадать с доказательством
        if total_signals > 0:
            evidence_rate = positive_signals / total_signals
            if evidence_rate > prior_mu and posterior_mu < prior_mu_calc:
                posterior_mu = prior_mu_calc + abs(posterior_mu - prior_mu_calc)
            elif evidence_rate < prior_mu and posterior_mu > prior_mu_calc:
                posterior_mu = prior_mu_calc - abs(posterior_mu - prior_mu_calc)

        posterior_mu = max(min(posterior_mu, 0.9999), 0.0001)

        # ── 80% CI через ppf апостериорного Beta-распределения ─────────────
        post_dist  = stats.beta(alpha_post, beta_post)
        ci_80_low  = round(post_dist.ppf(0.10) * 100, 1)
        ci_80_high = round(post_dist.ppf(0.90) * 100, 1)

        return BayesResult(
            prior_pct=round(prior_mu_calc * 100, 1),
            posterior_pct=round(posterior_mu * 100, 1),
            ci_80_low=ci_80_low,
            ci_80_high=ci_80_high,
        )

    def bayesian_contextual_risk(
        self,
        prior_error_rate: float,
        prob_condition_given_error: float,
        prob_condition: float,
    ) -> float:
        """
        Вычисляет условную вероятность ошибки при наличии условия (теорема Байеса).

        P(Ошибка|Условие) = P(Условие|Ошибка) * P(Ошибка) / P(Условие)

        Параметры:
            prior_error_rate: базовая вероятность ошибки P(E)
            prob_condition_given_error: вероятность условия при ошибке P(C|E)
            prob_condition: общая вероятность условия P(C)

        Возвращает:
            float: вероятность ошибки при наличии условия, ограниченная 1.0
        """
        if prob_condition == 0:
            return 1.0
        result = (prob_condition_given_error * prior_error_rate) / prob_condition
        return min(result, 1.0)


if __name__ == "__main__":
    engine = MathEngine()

    print("=== Тест graph_bottleneck ===")
    edges = [
        ("A", "B", 3.0),
        ("B", "C", 2.0),
        ("A", "C", 1.0),
        ("C", "D", 4.0),
        ("B", "D", 1.5),
    ]
    gr = engine.graph_bottleneck(edges)
    print(f"Узкое место: {gr.bottleneck_node} (оценка: {gr.bottleneck_score})")
    print(f"Рейтинг узлов: {gr.all_nodes_ranked}")

    print("\n=== Тест markov_absorbing ===")
    Q = np.array([[0.2, 0.3], [0.1, 0.4]])
    state_times = np.array([8.0, 16.0])
    states = ["Квалификация", "Предложение"]
    mr = engine.markov_absorbing(Q, state_times, states)
    print(f"Ожидаемое время: {mr.expected_lead_time_hours:.2f} ч")
    print(f"P(завершение): {mr.p_complete}, P(потеря): {mr.p_lost}")

    print("\n=== Тест bayesian_update ===")
    br = engine.bayesian_update(positive_signals=42, total_signals=120, prior_rate=0.34)
    print(f"Априорная: {br.prior_pct}%, Апостериорная: {br.posterior_pct}%")
    print(f"80% ДИ: [{br.ci_80_low}%, {br.ci_80_high}%]")

    print("\n=== Тест bayesian_contextual_risk ===")
    risk = engine.bayesian_contextual_risk(
        prior_error_rate=0.05,
        prob_condition_given_error=0.80,
        prob_condition=0.20,
    )
    print(f"P(Ошибка|Условие): {risk:.4f}")
