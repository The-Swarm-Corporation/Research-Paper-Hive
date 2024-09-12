import os
from swarms import Agent, OpenAIChat
from loguru import logger
from paper_hive.paper_fetcher import DailyPapersDownloader
from dotenv import load_dotenv

load_dotenv()

# Define a custom, extensive system prompt for AI research paper summarization
AI_RESEARCH_AGENT_SYS_PROMPT = """
You are an AI research paper summarization expert. Your task is to analyze and summarize AI research papers in a clear, concise, and detailed manner. For every paper you summarize, consider the following key points:

1. **Thoroughness**: Provide a summary that fully encapsulates the essence of the research. Highlight key contributions, novel methodologies, experiments, and findings.
2. **Technical Depth**: Dive into the technical aspects, explaining methodologies such as architectures, algorithms, and mathematical frameworks.
3. **Contextual Understanding**: Discuss the significance of the research in the broader AI field, including how it advances the state of the art or opens up new avenues for further research.
4. **Applications and Impact**: Where applicable, describe potential real-world applications and societal impacts of the research.
5. **Clarity**: Avoid jargon-heavy language when unnecessary. Make the summary accessible to a technically literate but non-expert audience.

When summarizing, make sure to:
- Start with a high-level overview (1-2 sentences) that describes the purpose of the paper.
- Summarize key sections of the paper, including methodology, experiments, results, and conclusions.
- Include any significant challenges, limitations, or future work mentioned by the authors.
- Maintain accuracy, and ensure that the summary is precise and avoids misinterpretation of the content.

You should provide concise yet comprehensive summaries that are highly reliable for a wide range of AI practitioners, including researchers, engineers, and students.

Be responsive and generate detailed, precise outputs with deep insights from the paper. Your summaries should be suitable for inclusion in a research digest or a presentation for colleagues in the AI field.
"""

# OpenAI API key setup
api_key = os.getenv("OPENAI_API_KEY")

# Create the OpenAIChat model instance
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize the agent with a custom system prompt for summarizing AI research papers
agent = Agent(
    agent_name="AI-Research-Paper-Summarizer",
    system_prompt=AI_RESEARCH_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="ai_research_summarizer.json",
    user_name="research_team",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)


def summarize_papers() -> str:
    # Example paper to summarize
    fetcher = DailyPapersDownloader()
    data = fetcher.download_daily_papers()
    data = fetcher.generate_markdown_string(data)

    prompt = f"""
    Please summarize the following AI research papers with accuracy and precision:
    
    {data}
    """

    # Run the agent with the prompt
    logger.info("Summarizing AI research paper...")
    try:
        summary = agent.run(prompt)
        logger.info("Summary generated successfully.")
        return summary
    except Exception as e:
        logger.error(
            f"An error occurred while generating the summary: {e}"
        )
        raise e
