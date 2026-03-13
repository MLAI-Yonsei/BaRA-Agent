import os
from typing import Optional, Tuple

from web_crawler.pipeline.runtime import run_step1


class Step1Manager:
    def __init__(self, config):
        self.config = config

    async def execute(self, api_key: Optional[str]) -> Tuple[bool, Optional[str]]:
        first_url = self.config.first_url

        if first_url and first_url.startswith(self.config.skip_step1_prefix):
            attachment_path = self.config.step1_links_path
            print(f"⏭️ Skipping Step 1: first_url starts with '{self.config.skip_step1_prefix}'.")
            print(f"📁 Using existing file: {attachment_path}")
            if not os.path.exists(attachment_path):
                raise FileNotFoundError(f"links_bfs.json file not found: {attachment_path}")
            return True, attachment_path

        result = await run_step1(
            self.config.wanted,
            first_url,
            api_key,
            self.config.model_name,
            self.config.max_depth,
            self.config.max_width,
            self.config.max_pages,
            max_attempts=self.config.max_attempts,
            llm_provider=self.config.llm_provider,
            ollama_host=self.config.ollama_host,
            ollama_api_key=self.config.ollama_api_key,
        )

        if not result:
            return False, None

        _, attachment_path = result
        print(f"📁 attachment_path: {attachment_path}")
        return True, attachment_path
