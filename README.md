# Auto Analyst - Agentic AI Automated Data Analyst

## Overview

**Auto Analyst** is an agentic AI application designed to empower non-technical users and business teams to effortlessly understand their data—whether from CSV files, Excel spreadsheets, or other tabular formats. Leveraging a modular team of specialized AI agents, Auto Analyst automates data loading, exploratory data analysis (EDA), insight generation, and report writing to deliver clear, actionable summaries in plain English.

## Real-World Problem

Many business users face challenges interpreting raw data due to limited technical expertise or time constraints, which slows decision-making and burdens data science teams. Auto Analyst addresses this by using **agentic AI** to fully automate the data analysis workflow, enabling fast, reliable, and interpretable insights without manual intervention.

## Agent Roles

- **Data Loader:** Parses and validates uploaded CSV or Excel files, ensuring data integrity and quality.  
- **Statistical Analyst:** Performs exploratory data analysis and computes key summary statistics.  
- **Insight Generator:** Detects meaningful patterns, trends, outliers, and correlations within the data.  
- **Report Writer:** Produces executive summaries written in clear, plain English tailored for business stakeholders.

Each agent specializes in a distinct phase of the data analysis pipeline and collaborates seamlessly through CrewAI’s agent orchestration framework.

## Technology Stack

Auto Analyst is built on top of **CrewAI**, a lightweight and modular Python framework for creating **agentic AI systems**. CrewAI enables you to define AI agents with explicit roles, goals, and backstories, facilitating transparent multi-agent collaboration and workflow orchestration.

The project integrates **Ollama**, a local large language model (LLM) platform that runs powerful language models entirely on your own machine. This local deployment eliminates the need for cloud-based APIs, ensuring **data security and privacy** by keeping sensitive information on-device, while delivering low latency and powerful natural language capabilities.

## Features & Benefits

- **Agentic AI Automation:** Autonomous agents collaborate to reason and execute complex workflows without manual scripting.  
- **Local LLM Integration:** Running Ollama locally ensures sensitive data never leaves your machine, maximizing privacy and security.  
- **Modular & Extensible:** Easily customize or extend agents to add new analysis capabilities or support other LLM backends.  
- **Business-Ready Reports:** Translates raw data insights into clear, actionable summaries accessible to non-technical audiences.  
- **Fast & Scalable:** Automates repetitive and time-consuming data tasks to accelerate decision-making cycles.

## How It Works

1. The **Data Loader Agent** ingests and validates the dataset.  
2. The **Statistical Analyst Agent** computes descriptive statistics and key metrics.  
3. The **Insight Generator Agent** identifies patterns, outliers, and correlations.  
4. The **Report Writer Agent** crafts a comprehensive summary in plain English.  
5. CrewAI orchestrates the workflow by invoking each agent sequentially and managing context passing.

## Getting Started

Clone the repository, set up a Python environment, install dependencies, and ensure Ollama is running locally at `http://localhost:11434`. Then, run the main script to analyze your dataset and receive a detailed summary report—fully automated by the agent team.

## Why Agentic AI?

Agentic AI systems like Auto Analyst go beyond simple prompt-response interactions by enabling multiple specialized AI agents to work collaboratively and autonomously on complex problems. This approach provides:

- End-to-end automation of workflows  
- Clear role delegation and specialization  
- Greater modularity and scalability  
- Stronger alignment with real-world business processes
