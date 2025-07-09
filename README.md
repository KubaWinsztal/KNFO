import pandas as pd
import numpy as np
from typing import List, Sequence

def most_prudent_pd_bounds_df(
    counts: pd.DataFrame,
    gamma: float = 0.90,
    grade_order: Sequence[str] = (
        "1", "2+", "2", "2-",
        "3+", "3", "3-",
        "4+", "4", "4-",
        "5+", "5", "5-",
        "6+", "6", "6-",
        "7+", "7", "7-",
    ),
) -> pd.DataFrame:
    """
    Zwraca górne granice PD (Section 2 Pluto–Tasche) dla każdego ratingu i okresu.

    Parametry
    ---------
    counts      : DataFrame — wiersze = ratingi, kolumny = okresy, wartości = liczba ekspozycji
    gamma       : poziom ufności (np. 0.90)
    grade_order : kolejność ratingów od najlepszego do najgorszego

    Wynik
    -----
    DataFrame o tym samym kształcie co *counts* z PD-ami (float 0–1)
    """
    if not (0 < gamma < 1):
        raise ValueError("gamma musi być z przedziału (0, 1).")

    # Upewnij się, że wiersze są w prawidłowej kolejności; brakujące ratingi = 0
    df = counts.reindex(index=grade_order).fillna(0)

    res = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        freqs = df[col].astype(int).to_numpy()
        # skumulowane od najgorszej klasy
        tail_counts = np.cumsum(freqs[::-1])[::-1]

        # formuła sekcji 2 z dodatkową regułą “zero → 1”
        p_max = np.where(
            freqs == 0,
            1.0,
            1.0 - (1.0 - gamma) ** (1.0 / tail_counts)
        )
        res[col] = p_max

    return res


# ─── Przyklad ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # przykładowy DataFrame: trzy kwartały
    data = {
        "2024Q1": [50, 60, 120, 70, 150, 180, 160, 200, 190, 180,
                   210, 230, 220, 80, 70, 30, 0, 0, 0],
        "2024Q2": [55, 58, 125, 68, 152, 175, 165, 205, 195, 185,
                   0, 0, 0, 0, 0, 0, 0, 0, 0],          # tylko część ratingów
        "2024Q3": [0] * 19                                # brak ekspozycji
    }
    grades = [
        "1", "2+", "2", "2-", "3+", "3", "3-",
        "4+", "4", "4-", "5+", "5", "5-",
        "6+", "6", "6-", "7+", "7", "7-"
    ]
    counts_df = pd.DataFrame(data, index=grades)

    pd_bounds = most_prudent_pd_bounds_df(counts_df, gamma=0.95)
    print(pd_bounds.head())