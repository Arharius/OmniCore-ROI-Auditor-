from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd


ABSORBING_KEYWORDS = {
    "approved", "closed", "завершён", "закрыт",
    "rejected", "отказ", "cancelled", "отменён",
    "lost", "потерян",
}


@dataclass
class ProcessLog:
    matrix_Q: np.ndarray
    states_all: list
    states_transient: list
    absorbing_states: list
    avg_time_per_state: dict
    raw_counts: dict
    total_deals: int
    error_rate: float
    avg_cycle_days: float
    avg_deal_value: float


class MatrixExtractor:
    """Экстрактор матриц переходов из журналов сделок для марковского анализа."""

    def _is_absorbing_by_keyword(self, state: str) -> bool:
        """Проверяет, является ли состояние поглощающим по ключевым словам."""
        return state.strip().lower() in ABSORBING_KEYWORDS

    def _build_process_log(
        self,
        sequences: dict,
        timestamps: dict,
        has_error_map: dict,
        deal_values: dict,
        cycle_days: list,
        error_rows: int = 0,
        total_rows: int = 0,
    ) -> ProcessLog:
        """
        Строит ProcessLog из извлечённых последовательностей переходов.

        Параметры:
            sequences: словарь {deal_id: [статус1, статус2, ...]}
            timestamps: словарь {deal_id: [timestamp1, timestamp2, ...]}
            has_error_map: словарь {deal_id: bool}
            deal_values: словарь {deal_id: float}
            cycle_days: список длительностей циклов в днях

        Возвращает:
            ProcessLog с матрицей Q и сводными метриками.
        """
        transition_counts = defaultdict(lambda: defaultdict(int))
        time_in_state = defaultdict(list)

        for deal_id, statuses in sequences.items():
            times = timestamps.get(deal_id, [])
            for i in range(len(statuses) - 1):
                frm = statuses[i]
                to = statuses[i + 1]
                transition_counts[frm][to] += 1
                if i < len(times) - 1 and times[i] is not None and times[i + 1] is not None:
                    delta_hours = (times[i + 1] - times[i]).total_seconds() / 3600
                    if delta_hours >= 0:
                        time_in_state[frm].append(delta_hours)

        all_states = sorted(
            set(s for seq in sequences.values() for s in seq)
        )

        outgoing = {s: sum(transition_counts[s].values()) for s in all_states}

        absorbing_states = sorted([
            s for s in all_states
            if self._is_absorbing_by_keyword(s) or outgoing.get(s, 0) == 0
        ])
        transient_states = sorted([s for s in all_states if s not in absorbing_states])

        n = len(transient_states)
        idx = {s: i for i, s in enumerate(transient_states)}
        Q = np.zeros((n, n))

        for frm in transient_states:
            row_total = sum(
                cnt for to, cnt in transition_counts[frm].items()
                if to in idx
            )
            if row_total > 0:
                for to, cnt in transition_counts[frm].items():
                    if to in idx:
                        Q[idx[frm]][idx[to]] = cnt / row_total
            else:
                Q[idx[frm]][idx[frm]] = 1.0

        raw_counts = {
            frm: dict(tos)
            for frm, tos in transition_counts.items()
        }

        avg_time_per_state = {
            s: round(float(np.mean(times)), 4) if times else 0.0
            for s, times in time_in_state.items()
        }

        total_deals = len(sequences)
        if total_rows > 0:
            error_rate = error_rows / total_rows
        else:
            error_count = sum(1 for v in has_error_map.values() if v)
            error_rate = error_count / total_deals if total_deals > 0 else 0.0

        avg_cycle_days = float(np.mean(cycle_days)) if cycle_days else 0.0

        values = [v for v in deal_values.values() if v is not None]
        avg_deal_value = float(np.mean(values)) if values else 0.0

        return ProcessLog(
            matrix_Q=Q,
            states_all=all_states,
            states_transient=transient_states,
            absorbing_states=absorbing_states,
            avg_time_per_state=avg_time_per_state,
            raw_counts=raw_counts,
            total_deals=total_deals,
            error_rate=round(error_rate, 4),
            avg_cycle_days=round(avg_cycle_days, 2),
            avg_deal_value=round(avg_deal_value, 2),
        )

    @staticmethod
    def _clean_state(name: str) -> str:
        """Убирает пометки в скобках: 'Development (Возврат!)' → 'Development'."""
        import re
        return re.sub(r"\s*\(.*?\)\s*$", "", str(name)).strip()

    @staticmethod
    def _detect_format(columns: list) -> str:
        """
        Определяет формат CSV по именам колонок.
        Возвращает 'transition_log' или 'deal_timeline'.
        """
        cols_lower = [c.lower() for c in columns]
        has_stage    = any("stage" in c for c in cols_lower)
        has_next     = any("next" in c for c in cols_lower)
        has_deal_id  = any(c in ("deal_id",) for c in cols_lower)
        has_status   = any(c == "status" for c in cols_lower)
        if has_stage and has_next and not (has_deal_id and has_status):
            return "transition_log"
        return "deal_timeline"

    def _from_transition_log(self, df: pd.DataFrame) -> ProcessLog:
        """
        Парсит формат: Task_ID | Stage | Time_Spent (дни) | Next_Stage.
        Колонки определяются по частичному совпадению имён.
        """
        cols = df.columns.tolist()

        def find_col(keywords):
            for k in keywords:
                for c in cols:
                    if k.lower() in c.lower():
                        return c
            return None

        col_id    = find_col(["task_id", "deal_id", "id"])
        col_stage = find_col(["stage", "status", "состояние"])
        col_time  = find_col(["time_spent", "time", "дней", "days", "duration"])
        col_next  = find_col(["next_stage", "next", "переход", "transition"])

        if not col_id or not col_stage or not col_next:
            raise ValueError(f"Не найдены нужные колонки. Есть: {cols}")

        sequences  = {}
        time_spent = {}
        cycle_days = []

        for task_id, group in df.groupby(col_id):
            group = group.reset_index(drop=True)
            stages = [self._clean_state(s) for s in group[col_stage].tolist()]
            nexts  = [self._clean_state(s) for s in group[col_next].tolist()]

            full_seq = stages[:1]
            for nxt in nexts:
                full_seq.append(nxt)
            sequences[task_id] = full_seq

            if col_time:
                days_list = pd.to_numeric(group[col_time], errors="coerce").fillna(0).tolist()
                time_spent[task_id] = days_list
                total_days = sum(days_list)
                if total_days > 0:
                    cycle_days.append(total_days)

        from collections import defaultdict
        transition_counts = defaultdict(lambda: defaultdict(int))
        time_in_state     = defaultdict(list)

        for task_id, seq in sequences.items():
            days = time_spent.get(task_id, [])
            for i in range(len(seq) - 1):
                frm = seq[i]
                to  = seq[i + 1]
                transition_counts[frm][to] += 1
                if i < len(days):
                    time_in_state[frm].append(days[i] * 24)

        all_states = sorted(set(s for seq in sequences.values() for s in seq))
        outgoing   = {s: sum(transition_counts[s].values()) for s in all_states}

        absorbing_states = sorted([
            s for s in all_states
            if self._is_absorbing_by_keyword(s) or outgoing.get(s, 0) == 0
        ])
        transient_states = sorted([s for s in all_states if s not in absorbing_states])

        n   = len(transient_states)
        idx = {s: i for i, s in enumerate(transient_states)}
        Q   = np.zeros((n, n))

        for frm in transient_states:
            row_total = sum(
                cnt for to, cnt in transition_counts[frm].items() if to in idx
            )
            if row_total > 0:
                for to, cnt in transition_counts[frm].items():
                    if to in idx:
                        Q[idx[frm]][idx[to]] = cnt / row_total
            else:
                Q[idx[frm]][idx[frm]] = 1.0

        avg_time_per_state = {
            s: round(float(np.mean(times)), 4) if times else 0.0
            for s, times in time_in_state.items()
        }

        total_deals = len(sequences)
        avg_cycle   = float(np.mean(cycle_days)) if cycle_days else 0.0

        return ProcessLog(
            matrix_Q=Q,
            states_all=all_states,
            states_transient=transient_states,
            absorbing_states=absorbing_states,
            avg_time_per_state=avg_time_per_state,
            raw_counts={frm: dict(tos) for frm, tos in transition_counts.items()},
            total_deals=total_deals,
            error_rate=0.0,
            avg_cycle_days=round(avg_cycle, 2),
            avg_deal_value=0.0,
        )

    def from_csv(self, filepath: str) -> ProcessLog:
        """
        Читает CSV и автоматически определяет формат:
        - transition_log: Task_ID | Stage | Time_Spent | Next_Stage
        - deal_timeline:  Deal_ID | Status | Timestamp | Has_Error | Deal_Value
        """
        try:
            df  = pd.read_csv(filepath)
            fmt = self._detect_format(df.columns.tolist())
            print(f"[MatrixExtractor.from_csv] Формат: {fmt}")

            if fmt == "transition_log":
                return self._from_transition_log(df)

            df = pd.read_csv(filepath, parse_dates=["Timestamp"])
            df = df.sort_values(["Deal_ID", "Timestamp"]).reset_index(drop=True)

            sequences = {}
            timestamps = {}
            has_error_map = {}
            deal_values = {}
            cycle_days = []
            total_rows = 0
            error_rows = 0

            for deal_id, group in df.groupby("Deal_ID"):
                statuses = group["Status"].tolist()
                times = group["Timestamp"].tolist()
                sequences[deal_id] = statuses
                timestamps[deal_id] = times

                errors = group["Has_Error"].astype(str).str.lower()
                error_flags = [e in ("true", "1", "yes") for e in errors]
                has_error_map[deal_id] = any(error_flags)
                total_rows += len(error_flags)
                error_rows += sum(error_flags)

                vals = group["Deal_Value"].dropna()
                deal_values[deal_id] = float(vals.mean()) if not vals.empty else None

                if len(times) >= 2 and pd.notna(times[0]) and pd.notna(times[-1]):
                    delta = (times[-1] - times[0]).total_seconds() / 86400
                    cycle_days.append(max(delta, 0))

            return self._build_process_log(
                sequences, timestamps, has_error_map, deal_values, cycle_days,
                error_rows=error_rows, total_rows=total_rows,
            )

        except Exception as e:
            print(f"[MatrixExtractor.from_csv] Ошибка: {e}")
            return ProcessLog(
                matrix_Q=np.zeros((1, 1)),
                states_all=[],
                states_transient=[],
                absorbing_states=[],
                avg_time_per_state={},
                raw_counts={},
                total_deals=0,
                error_rate=0.0,
                avg_cycle_days=0.0,
                avg_deal_value=0.0,
            )

    def from_dict(self, records: list) -> ProcessLog:
        """
        Строит ProcessLog из списка словарей (ручной ввод).

        Каждый словарь должен содержать ключи:
        Deal_ID, Status, Timestamp (str или datetime), Has_Error, Deal_Value.

        Параметры:
            records: список словарей с данными сделок

        Возвращает:
            ProcessLog с матрицей переходов и агрегированными метриками.

        Исключения:
            Перехватывает все ошибки и возвращает пустой ProcessLog при сбое.
        """
        try:
            df = pd.DataFrame(records)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.sort_values(["Deal_ID", "Timestamp"]).reset_index(drop=True)

            sequences = {}
            timestamps = {}
            has_error_map = {}
            deal_values = {}
            cycle_days = []
            total_rows = 0
            error_rows = 0

            for deal_id, group in df.groupby("Deal_ID"):
                statuses = group["Status"].tolist()
                times = group["Timestamp"].tolist()
                sequences[deal_id] = statuses
                timestamps[deal_id] = times

                errors = group["Has_Error"].astype(str).str.lower()
                error_flags = [e in ("true", "1", "yes") for e in errors]
                has_error_map[deal_id] = any(error_flags)
                total_rows += len(error_flags)
                error_rows += sum(error_flags)

                vals = group["Deal_Value"].dropna()
                deal_values[deal_id] = float(vals.mean()) if not vals.empty else None

                if len(times) >= 2 and pd.notna(times[0]) and pd.notna(times[-1]):
                    delta = (times[-1] - times[0]).total_seconds() / 86400
                    cycle_days.append(max(delta, 0))

            return self._build_process_log(
                sequences, timestamps, has_error_map, deal_values, cycle_days,
                error_rows=error_rows, total_rows=total_rows,
            )

        except Exception as e:
            print(f"[MatrixExtractor.from_dict] Ошибка: {e}")
            return ProcessLog(
                matrix_Q=np.zeros((1, 1)),
                states_all=[],
                states_transient=[],
                absorbing_states=[],
                avg_time_per_state={},
                raw_counts={},
                total_deals=0,
                error_rate=0.0,
                avg_cycle_days=0.0,
                avg_deal_value=0.0,
            )


if __name__ == "__main__":
    records = [
        {"Deal_ID": 1, "Status": "New",        "Timestamp": "2024-01-01 09:00", "Has_Error": False, "Deal_Value": 500},
        {"Deal_ID": 1, "Status": "Qualified",   "Timestamp": "2024-01-03 10:00", "Has_Error": False, "Deal_Value": 500},
        {"Deal_ID": 1, "Status": "Proposal",    "Timestamp": "2024-01-07 12:00", "Has_Error": True,  "Deal_Value": 500},
        {"Deal_ID": 1, "Status": "closed",      "Timestamp": "2024-01-10 15:00", "Has_Error": False, "Deal_Value": 500},

        {"Deal_ID": 2, "Status": "New",         "Timestamp": "2024-01-02 08:00", "Has_Error": False, "Deal_Value": 800},
        {"Deal_ID": 2, "Status": "Qualified",   "Timestamp": "2024-01-04 09:00", "Has_Error": False, "Deal_Value": 800},
        {"Deal_ID": 2, "Status": "rejected",    "Timestamp": "2024-01-06 11:00", "Has_Error": False, "Deal_Value": 800},

        {"Deal_ID": 3, "Status": "New",         "Timestamp": "2024-01-05 10:00", "Has_Error": True,  "Deal_Value": 650},
        {"Deal_ID": 3, "Status": "Proposal",    "Timestamp": "2024-01-09 14:00", "Has_Error": False, "Deal_Value": 650},
        {"Deal_ID": 3, "Status": "approved",    "Timestamp": "2024-01-15 16:00", "Has_Error": False, "Deal_Value": 650},
    ]

    extractor = MatrixExtractor()
    log = extractor.from_dict(records)

    print("=== ProcessLog ===")
    print(f"Всего сделок   : {log.total_deals}")
    print(f"Все состояния  : {log.states_all}")
    print(f"Поглощающие    : {log.absorbing_states}")
    print(f"Переходные     : {log.states_transient}")
    print(f"Доля ошибок    : {log.error_rate:.2%}")
    print(f"Средний цикл   : {log.avg_cycle_days} дн.")
    print(f"Средняя сделка : {log.avg_deal_value} EUR")
    print(f"Матрица Q ({log.matrix_Q.shape}):")
    print(log.matrix_Q)
    print(f"Среднее время по состояниям: {log.avg_time_per_state}")
    print(f"Счётчики переходов: {log.raw_counts}")
