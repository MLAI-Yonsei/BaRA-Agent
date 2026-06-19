from web_crawler.pipeline.runtime import run_step2


class Step2Manager:
    def __init__(self, config):
        self.config = config

    async def iterate(self, api_key, attachment_path):
        async for sub_url, last_extracted_content, page_index in run_step2(
            self.config.wanted,
            api_key,
            self.config.model_name,
            max_attempts=self.config.max_attempts,
            start_url=attachment_path,
            first_url=self.config.first_url,
            firecrawl_api_key=self.config.firecrawl_api_key,
            llm_provider=self.config.llm_provider,
            ollama_host=self.config.ollama_host,
            ollama_api_key=self.config.ollama_api_key,
            step2_union_retry_attempts=self.config.step2_union_retry_attempts,
            ablation_no_reflection=self.config.ablation_no_reflection,
            ablation_retry_merge_mode=self.config.ablation_retry_merge_mode,
            step2_model_name=self.config.step2_model_name,
            step2_concurrency=self.config.step2_concurrency,
        ):
            yield sub_url, last_extracted_content, page_index
