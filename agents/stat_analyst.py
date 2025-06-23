from crewai.agent import Agent
import pandas as pd
import numpy as np

class StatisticalAnalyst(Agent):
    def __init__(self, llm):
        super().__init__(
        name="Data Loader",
        role= "Statistical Analyst",
        goal= "Perform exploratory data analysis and generate summary statistics.",
        backstory= 
            """
            As a statistical analyst, I rigorously analyze datasets to uncover 
            key metrics, distributions, missing values, and correlations that 
            help business users understand their data.
            """,

        llm=llm
        )

    def analyze(self, df: pd.DataFrame) -> str:
        summary = "**Exploratory Data Analysis Summary**\n\n"
        summary += f"Total rows: {len(df)}\n"
        summary += f"Total columns: {len(df.columns)}\n\n"

        # Data types
        summary += "**Column Types:**\n"
        summary += df.dtypes.apply(lambda x: f"- {x.name}").to_string()
        summary += "\n\n"

        # Unique values
        summary += "**Unique Values per Column:**\n"
        unique_counts = df.nunique()
        for col, count in unique_counts.items():
            summary += f"- {col}: {count} unique values\n"
        summary += "\n"

        # Missing data
        summary += "**Missing Data:**\n"
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        for col in df.columns:
            if missing_counts[col] > 0:
                summary += f"- {col}: {missing_counts[col]} missing ({missing_pct[col]:.2f}%)\n"
        total_cells = df.shape[0] * df.shape[1]
        empty_cells = df.isnull().sum().sum()
        completeness = 100 * (1 - empty_cells / total_cells)
        summary += f"Overall dataset completeness: {completeness:.2f}%\n\n"

        # Descriptive statistics
        desc = df.describe(include='all').transpose()
        summary += "**Descriptive Statistics:**\n"
        for col, stats in desc.iterrows():
            summary += f"- {col}:\n"
            for stat, val in stats.items():
                summary += f"    {stat}: {val}\n"
        summary += "\n"

        # Constant or zero-variance columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            summary += "**Constant Columns (zero variance):**\n"
            for col in constant_cols:
                summary += f"- {col}\n"
            summary += "\n"

        # Frequent values for categoricals
        summary += "**Top Frequencies (Categoricals):**\n"
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            summary += f"- {col}:\n"
            top_vals = df[col].value_counts(normalize=True).head(3)
            for val, pct in top_vals.items():
                summary += f"    {val}: {pct*100:.1f}%\n"
        summary += "\n"

        # Numeric analysis
        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            summary += "**Numerical Column Analysis:**\n"
            for col in numeric_df.columns:
                skew = numeric_df[col].skew()
                kurt = numeric_df[col].kurtosis()
                summary += f"- {col}: skew={skew:.2f}, kurtosis={kurt:.2f}\n"
            summary += "\n"

            # Outlier detection using IQR
            summary += "**Potential Outliers (IQR method):**\n"
            for col in numeric_df.columns:
                q1 = numeric_df[col].quantile(0.25)
                q3 = numeric_df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)]
                pct_outliers = 100 * len(outliers) / len(numeric_df)
                summary += f"- {col}: {pct_outliers:.1f}% values are outliers\n"
            summary += "\n"

            # Correlation matrix
            summary += "**Top Correlations Between Numeric Columns:**\n"
            corr = numeric_df.corr().abs()
            corr_pairs = corr.where(~np.tril(np.ones(corr.shape)).astype(bool)).stack()
            top_corrs = corr_pairs.sort_values(ascending=False).head(5)
            for (col1, col2), val in top_corrs.items():
                summary += f"- {col1} ↔ {col2}: {val:.2f}\n"
            summary += "\n"

            prompt = (
                "You're a data analyst. Given the following EDA summary, provide a clear, structured, "
                "non-technical explanation of the dataset's key characteristics, including missing data, "
                "important distributions, and any red flags. Make it readable for business stakeholders.\n\n"
                f"{summary}"
            )

            try:
                response = self.llm.call(prompt)
                return f"**LLM Analysis Summary:**\n{response.strip()}"
            except Exception as e:
                return f"{summary}\n\nLLM failed to summarize: {str(e)}"

