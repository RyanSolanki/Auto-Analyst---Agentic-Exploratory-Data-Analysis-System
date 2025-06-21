import pandas as pd
from agents.data_loader import DataLoader
# from agents.statistical_analyst import StatisticalAnalystAgent
# from agents.insight_generator import InsightGeneratorAgent
# from agents.report_writer import ReportWriterAgent
from models.ollama_wrapper import OllamaLLM

def main(data_path: str):
    # Initialize your LLM wrapper (Ollama local LLM)
    llm = OllamaLLM(model_name="llama3")

    # Step 1: Load and validate data
    data_loader = DataLoader(llm=llm)
    df = data_loader.load_and_describe(data_path)

    # # Step 2: Analyze data with Statistical Analyst
    # analyst = StatisticalAnalyst(llm=llm)
    # eda_summary = analyst.analyze(df)
    # print("EDA Summary:\n", eda_summary)

    # # Step 3: Generate insights based on EDA
    # insight_gen = InsightGenerator(llm=llm)
    # insights = insight_gen.generate_insights(eda_summary)
    # print("Insights:\n", insights)

    # # Step 4: Write final report
    # report_writer = ReportWriter(llm=llm)
    # report = report_writer.write_report(insights)
    # print("Final Report:\n", report)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_csv>")
        sys.exit(1)

    data_file = sys.argv[1]
    main(data_file)