import pandas as pd
from agents.data_loader import DataLoader
from agents.stat_analyst import StatisticalAnalyst
from agents.insights_generator import InsightGenerator
# from agents.report_writer import ReportWriter
from models.ollama_wrapper import OllamaLLM

def main(data_path: str):
    # Read in data to pandas df
    df = pd.read_csv(data_path)

    # Initialize LLM wrapper for local Ollama model
    llm = OllamaLLM(model_name="llama3")

    # Step 1: Load, validate, and summarize data
    data_loader = DataLoader(llm=llm)
    data_report = data_loader.load_and_describe(df)

    # Step 2: Analyze data with Statistical Analyst
    analyst = StatisticalAnalyst(llm=llm)
    data_report += analyst.analyze(df)

    # Step 3: Generate insights based on raw data, data summary, and EDA
    insight_gen = InsightGenerator(llm=llm)
    insights = insight_gen.generate_insights(df, data_report)
    print("Insights:\n", insights)

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