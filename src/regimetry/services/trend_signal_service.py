import pandas as pd

class TrendSignalService:
    @staticmethod
    def add_trend_signals(df: pd.DataFrame, threshold: float = 0.002) -> pd.DataFrame:
        """
        Adds binary trend signal columns for regime classification.

        Args:
            df (pd.DataFrame): DataFrame with AHMA, Leavitt_Projection, LC_Slope, etc.
            threshold (float): Used for future RHD variants (not applied here).

        Returns:
            pd.DataFrame: Updated DataFrame with trend signal columns.
        """
        # Compute LP_Slope from existing columns
        df["LP_Slope"] = df["Leavitt_Projection"] - df["LP_Prev"]

        ahma_shifted = df["AHMA"].shift(1)

        # Core conditions
        cond_lc_lp_bull = (df["LC_Slope"] > 0) & (df["LP_Slope"] > 0)
        cond_lc_lp_bear = (df["LC_Slope"] < 0) & (df["LP_Slope"] < 0)

        # Trend Confirmation
        df["Trend_Bull"] = (
            (df["AHMA"] > df["Leavitt_Convolution"]) &
            (df["AHMA"] > df["Leavitt_Projection"]) &
            cond_lc_lp_bull
        )

        df["Trend_Bear"] = (
            (df["AHMA"] < df["Leavitt_Convolution"]) &
            (df["AHMA"] < df["Leavitt_Projection"]) &
            cond_lc_lp_bear
        )

        # Expansion
        df["Expansion_Bull"] = (
            (df["LC_Slope"] > df["LP_Slope"]) & cond_lc_lp_bull
        )

        df["Expansion_Bear"] = (
            (df["LC_Slope"] < df["LP_Slope"]) & cond_lc_lp_bear
        )

        # Momentum Divergence
        df["Momentum_Divergence_Bull"] = (df["LC_Slope"] > 0) & (df["LP_Slope"] < 0)
        df["Momentum_Divergence_Bear"] = (df["LC_Slope"] < 0) & (df["LP_Slope"] > 0)

        # Classic Divergence
        df["Classic_Divergence_Bull"] = (df["AHMA"] < ahma_shifted) & (df["LC_Slope"] > 0)
        df["Classic_Divergence_Bear"] = (df["AHMA"] > ahma_shifted) & (df["LC_Slope"] < 0)

        # Reverse Hidden Divergence (Strong only)
        df["Strong_RHD_Bull"] = (df["AHMA"] < ahma_shifted) & cond_lc_lp_bull
        df["Strong_RHD_Bear"] = (df["AHMA"] > ahma_shifted) & cond_lc_lp_bear

        proximity = (df["Leavitt_Convolution"] - df["Leavitt_Projection"]).abs()

        df["Moderate_RHD_Bull"] = df["Strong_RHD_Bull"] & (proximity < 0.002)
        df["Moderate_RHD_Bear"] = df["Strong_RHD_Bear"] & (proximity < 0.002)

        df["Weak_RHD_Bull"] = df["Strong_RHD_Bull"] & (df["Leavitt_Convolution"] > df["Leavitt_Projection"])
        df["Weak_RHD_Bear"] = df["Strong_RHD_Bear"] & (df["Leavitt_Convolution"] < df["Leavitt_Projection"])

        return df
