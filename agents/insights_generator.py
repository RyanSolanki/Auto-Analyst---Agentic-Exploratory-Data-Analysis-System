from crewai import Agent
import pandas as pd

class InsightGenerator(Agent):
    role: str = "Insight Generator"
    goal: str = "Analyze data and EDA summaries to extract meaningful business insights."
    backstory: str = (
        "As an Insight Generator, I combine quantitative summaries with raw data context "
        "to highlight trends, anomalies, and relationships. I help business teams make informed decisions "
        "by translating data into plain-language insights."
    )

    def __init__(self, llm):
        super().__init__(llm=llm)
        self.llm = llm

    # Generate insights using both the EDA summary and raw dataset.
    def generate_insights(self, df: pd.DataFrame, eda_summary: str) -> str:
        # Extract schema-level stats
        row_count = len(df)
        col_count = len(df.columns)
        column_types = df.dtypes.apply(lambda x: x.name).to_dict()
        preview = df.head(5).to_markdown(index=False)

        # LLM prompt
        prompt = (
            "You are a senior data analyst. Your goal is to synthesize key insights "
            "from the following exploratory data analysis (EDA) summary and the raw dataset information. "
            "Look for patterns, trends, correlations, or anomalies that would be important to a business audience. "
            "Explain any implications clearly. Be precise, confident, and business-oriented.\n\n"
            "Dataset Overview:\n"
            f"- Total rows: {row_count}\n"
            f"- Total columns: {col_count}\n"
            f"- Column types: {column_types}\n\n"
            "First 5 rows of data:\n"
            f"{preview}\n\n"
            "EDA Summary:\n"
            f"{eda_summary}\n\n"
            "Please now return a concise but insightful analysis."
        )

        try:
            response = self.llm.call(prompt)
            return f"Generated Insights:\n\n{response.strip()}"
        except Exception as e:
            return f"Insight generation failed: {str(e)}"
