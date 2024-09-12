# Research-Paper-Hive

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


**Research-Paper-Hive** is an intelligent AI-powered application that helps you find and summarize research papers based on your preferences. Whether you're diving into a new research area or looking for papers tailored to specific topics, Research-Paper-Hive uses a swarm of AI agents to streamline your workflow by searching, analyzing, and summarizing relevant research papers for you.

## Features

- **Personalized Paper Search**: Input your preferences such as keywords, topics, or research fields, and Research-Paper-Hive will find relevant papers.
- **AI-Powered Summaries**: Agents collaborate to summarize each paper, providing you with concise and informative overviews.
- **Fast and Efficient**: With the power of swarm intelligence, Research-Paper-Hive processes and delivers results quickly.
- **Customizable Search Criteria**: Tailor your search by adjusting the specificity of your preferences.
- **Paper Ranking**: Get a ranked list of papers that are most aligned with your research interests.

## How It Works

1. **Input Preferences**: Provide Research-Paper-Hive with your specific preferences such as topics, keywords, or desired research fields.
2. **Agent Search**: A swarm of AI agents will search through academic databases to find the most relevant papers.
3. **Summarization**: Once papers are found, each agent works to generate concise summaries.
4. **Review and Download**: Review the summarized papers, ranked by relevance, and download the ones you need.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.10
- Required dependencies from `requirements.txt`

### Installation
```bash
$ pip3 install -U rph
```

### API Keys Setup
MedInsight Pro requires access to the OpenAI API, PubMed, and Semantic Scholar APIs. You’ll need to set up environment variables for these keys in your .env file:

```bash
OPENAI_API_KEY="your-openai-api-key"
WORKSPACE_ID="your-workspace-id" # Your workspace ID 
```


### Usage

```python

from rph.agent import summarize_papers

if __name__ == "__main__":
    summary = summarize_papers()
    print(summary)

```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
