from dataclasses import dataclass
import numpy as np
import networkx as nx
from scipy import stats


@dataclass
class GraphResult:
    bottleneck_node: str
    bottleneck_score: float
    pagerank: dict
    betweenness: dict
    all_nodes_ranked: list


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
        Выполняет байесовское обновление оценки вероятности успеха.

        Параметры:
            positive_signals: количество положительных сигналов
            total_signals: общее количество сигналов
            prior_rate: априорная вероятность (по умолчанию 0.34)

        Возвращает:
            BayesResult с априорной, апостериорной вероятностями и 80% доверительным интервалом.
        """
        alpha_prior = prior_rate * 10
        beta_prior = (1 - prior_rate) * 10

        alpha_post = alpha_prior + positive_signals
        beta_post = beta_prior + (total_signals - positive_signals)

        prior_dist = stats.beta(alpha_prior, beta_prior)
        post_dist = stats.beta(alpha_post, beta_post)

        prior_pct = round(prior_dist.mean() * 100, 1)
        posterior_pct = round(post_dist.mean() * 100, 1)
        ci_80_low = round(post_dist.ppf(0.10) * 100, 1)
        ci_80_high = round(post_dist.ppf(0.90) * 100, 1)

        return BayesResult(
            prior_pct=prior_pct,
            posterior_pct=posterior_pct,
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
