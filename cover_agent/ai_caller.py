import datetime
import os
import time

from functools import wraps
from typing import Optional
import yaml

import litellm

from tenacity import retry, stop_after_attempt, wait_fixed
from wandb.sdk.data_types.trace_tree import Trace

from cover_agent.custom_logger import CustomLogger
from cover_agent.record_replay_manager import RecordReplayManager
from cover_agent.settings.config_loader import get_settings
from cover_agent.utils import get_original_caller
import os
from dotenv import load_dotenv
load_dotenv()

headers = {
    "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}"
}
def conditional_retry(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.enable_retry:
            return func(self, *args, **kwargs)

        model_retries = get_settings().get("default").get("model_retries", 3)

        @retry(stop=stop_after_attempt(model_retries), wait=wait_fixed(1))
        def retry_wrapper():
            return func(self, *args, **kwargs)

        return retry_wrapper()

    return wrapper


class AICaller:
    def __init__(
        self,
        model: str,
        api_base: str = "",
        enable_retry=True,
        max_tokens=16384,  # TODO: Move to configuration.toml?
        source_file: str = None,
        test_file: str = None,
        record_mode: bool = False,
        record_replay_manager: Optional[RecordReplayManager] = None,
        logger: Optional[CustomLogger] = None,
        generate_log_files: bool = True,
    ):
        """
        Initializes an instance of the AICaller class.

        Parameters:
            model (str): The name of the model to be used.
            api_base (str): The base API URL to use in case the model is set to Ollama or Hugging Face.
        """
        self.model = model
        self.api_base = api_base
        self.enable_retry = enable_retry
        self.max_tokens = max_tokens
        self.source_file = source_file
        self.test_file = test_file
        self.record_mode = record_mode
        self.record_replay_manager = record_replay_manager or RecordReplayManager(
            record_mode=record_mode, generate_log_files=generate_log_files
        )
        self.logger = logger or CustomLogger.get_logger(__name__, generate_log_files=generate_log_files)

    @conditional_retry  # You can access self.enable_retry here
    def call_model(self, prompt: dict, stream=True):
        caller_name = get_original_caller()

        if "system" not in prompt or "user" not in prompt:
            raise KeyError("The prompt dictionary must contain 'system' and 'user' keys.")

        if prompt["system"] == "":
            messages = [{"role": "user", "content": prompt["user"]}]
        elif self.model in ["o1-preview", "o1-mini"]:
            messages = [{"role": "user", "content": prompt["system"] + "\n" + prompt["user"]}]
        else:
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]

        completion_params = {
            "model": os.getenv("LITELLM_MODEL"),
            "api_base": os.getenv("LITELLM_API_BASE"),
            "messages": messages,
            "stream": stream,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
            "logging": False,
            "provider": "ollama"
        }

        if self.model in ["o1-preview", "o1-mini", "o1", "o3-mini"]:
            stream = False
            completion_params.update({
                "temperature": 1,
                "stream": False,
                "max_completion_tokens": 2 * self.max_tokens,
            })
            completion_params.pop("max_tokens", None)

        if any(x in self.model for x in ["ollama", "huggingface"]) or self.model.startswith("openai/"):
            completion_params["api_base"] = self.api_base
            completion_params["provider"] = os.getenv("LITELLM_PROVIDER")

        try:
            self.logger.info(f"📣 Calling LLM from {caller_name}()...")
            response = litellm.completion(**completion_params)
        except Exception as e:
            self.logger.error(f"🔥 Error calling LLM model: {e}")
            raise e

        # ✅ Response handling
        try:
            if stream:
                # Ollama doesn’t support proper streaming; skip it
                stream = False
                self.logger.warning("⚠️ Forcing stream=False due to Ollama limitations")

            if hasattr(response, "choices"):
                content = response.choices[0].message.content
                usage = getattr(response, "usage", None)
            elif isinstance(response, dict):
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = None
            elif isinstance(response, str):
                content = response
                usage = None
            else:
                content = str(response)
                usage = None

            if isinstance(content, str):
                try:
                    content = yaml.safe_load(content)  # Attempt to parse as YAML
                    if not isinstance(content, dict):
                        raise ValueError("Parsed content is not a dictionary")
                except (yaml.YAMLError, ValueError) as e:
                    self.logger.warning(f"Content is not valid YAML or not a dictionary: {e}")
                    content = {"raw_response": content}  # Wrap raw response in a dictionary

            prompt_tokens = int(getattr(usage, "prompt_tokens", 0)) if usage else 0
            completion_tokens = int(getattr(usage, "completion_tokens", 0)) if usage else 0

            print("\n✅ Response:\n", content)
            self.logger.info("Printing results from LLM model...")

        except Exception as e:
            self.logger.error(f"🔥 Error parsing response: {e}")
            content = {"raw_response": str(response)}
            prompt_tokens = completion_tokens = 0

        # ✅ W&B logging
        if "WANDB_API_KEY" in os.environ:
            try:
                Trace(
                    name="inference_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    kind="llm",
                    inputs={"user_prompt": prompt["user"], "system_prompt": prompt["system"]},
                    outputs={"model_response": content},
                ).log(name="inference")
            except Exception as e:
                self.logger.error(f"⚠️ Error logging to W&B: {e}")

        # ✅ Replay/record
        if self.record_mode and self.source_file and self.test_file:
            self.record_replay_manager.record_response(
                self.source_file,
                self.test_file,
                prompt,
                content,
                prompt_tokens,
                completion_tokens,
                caller_name,
            )

        return content, prompt_tokens, completion_tokens
