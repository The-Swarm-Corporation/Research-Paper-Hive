import requests
import re
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, ValidationError
from loguru import logger


# Define the metadata model using Pydantic
class PaperMetadata(BaseModel):
    id: str
    title: str
    summary: str
    publishedAt: datetime


class DailyPapersDownloader:
    def __init__(self):
        # Set up logging with loguru
        logger.add(
            "daily_papers_log.log", rotation="500 MB"
        )  # Log file setup

    def download_daily_papers(
        self, date: Optional[str] = None
    ) -> Dict:
        """
        Downloads the daily papers JSON data for a specific date.

        Args:
            date (Optional[str]): The date in the format YYYYMMDD. If None, today's date is used.

        Returns:
            Dict: The downloaded data as a dictionary.

        Raises:
            ValueError: If the date is invalid.
            requests.RequestException: If the request to download the papers fails.
        """
        try:
            # If no date is provided, use today's date
            if date is None:
                date = datetime.now().strftime("%Y%m%d")

            # Validate date format
            if len(date) != 8 or not date.isdigit():
                raise ValueError(
                    "Invalid date format. Please use YYYYMMDD."
                )

            formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            url = f"https://huggingface.co/api/daily_papers?date={formatted_date}"

            # Download the file
            response = requests.get(url)
            response.raise_for_status()

            # Parse the downloaded content into a dictionary
            data = response.json()

            logger.info(
                f"Successfully downloaded daily papers for {formatted_date}"
            )
            return data

        except requests.RequestException as e:
            logger.error(f"Error downloading daily papers: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Cleans the text by removing newlines and extra spaces.

        Args:
            text (str): The text to clean.

        Returns:
            str: Cleaned text.
        """
        text = text.replace("\n", " ")
        return re.sub(r"\s+", " ", text).strip()

    def json_to_markdown(
        self, json_data: List[dict]
    ) -> Optional[str]:
        """
        Converts a list of paper metadata (in JSON format) to markdown format.

        Args:
            json_data (List[dict]): The list of papers metadata.

        Returns:
            Optional[str]: Markdown content or None if the data is empty or invalid.
        """
        if not json_data:
            return None

        try:
            # Validate the paper metadata using Pydantic
            papers = [
                PaperMetadata(**article["paper"])
                for article in json_data
            ]
        except ValidationError as e:
            logger.error(f"Error validating paper metadata: {e}")
            return None

        first_paper_date = papers[0].publishedAt.strftime("%Y-%m-%d")
        markdown_content = (
            f"# Daily Papers Summary for {first_paper_date}\n\n"
        )

        for paper in papers:
            title = self.clean_text(paper.title)
            summary = self.clean_text(paper.summary)
            hf_link = f"https://huggingface.co/papers/{paper.id}"
            arxiv_link = f"https://arxiv.org/pdf/{paper.id}"

            markdown_content += f"## {title}\n\n"
            markdown_content += f"[Open in Hugging Face]({hf_link}) | [Open PDF]({arxiv_link})\n\n"
            markdown_content += f"{summary}\n\n"

        return markdown_content

    def generate_markdown_string(
        self, json_data: Dict
    ) -> Optional[str]:
        """
        Converts a dictionary of papers directly into a markdown string.

        Args:
            json_data (Dict): The list of papers metadata in JSON format.

        Returns:
            Optional[str]: The generated markdown content as a string, or None if there's an error or no data.
        """
        if not isinstance(json_data, list) or len(json_data) == 0:
            logger.warning(
                "No valid JSON data provided for markdown generation."
            )
            return None

        markdown_output = self.json_to_markdown(json_data)

        if markdown_output:
            logger.info("Markdown string generated successfully.")
            return markdown_output
        else:
            logger.warning("No markdown content was generated.")
            return None


# if __name__ == "__main__":
#     downloader = DailyPapersDownloader()

#     # Download today's daily papers as a dictionary
#     data = downloader.download_daily_papers()

#     # Generate the markdown string directly from the JSON dictionary
#     markdown_string = downloader.generate_markdown_string(data)

#     if markdown_string:
#         print(markdown_string)  # Output the markdown string to the console
