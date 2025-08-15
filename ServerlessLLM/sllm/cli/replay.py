# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import json
import os
import time
import csv
from argparse import Namespace, _SubParsersAction

from openai import AsyncOpenAI

from sllm.cli._cli_utils import read_config
from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class ReplayCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        replay_parser = parser.add_parser(
            "replay", help="Replay requests based on workload and dataset."
        )
        replay_parser.add_argument(
            "--workload",
            type=str,
            required=True,
            help="Path to the CSV workload file",
        )
        replay_parser.add_argument(
            "--output",
            type=str,
            default="latency_results.json",
            help="Path to the output JSON file for latency results.",
        )
        replay_parser.set_defaults(func=ReplayCommand)

    def __init__(self, args: Namespace) -> None:
        self.workload_path = args.workload
        self.output_path = args.output
        self.url = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8343")
        self.base_url = self.url + "/v1"

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="API_KEY_PLACEHOLDER",  # Placeholder for API key
        )
        self.latency_results = []

    async def run(self) -> None:
        workload_rows = []
        try:
            with open(self.workload_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip the header
                for row in reader:
                    timestamp_str = row[0]
                    model_name = row[1]
                    input_len = int(row[2])
                    output_len = int(row[3])
                    h, m_s = timestamp_str.split(':', 1)
                    m, s_ms = m_s.split(':', 1)
                    s, ms = s_ms.split('.')
                    total_s = (
                        int(h) * 3600000 +
                        int(m) * 60000 +
                        int(s) * 1000 +
                        int(ms)
                    ) / 1000
                    workload_rows.append((model_name, total_s, input_len, output_len))
        except Exception as e:
            logger.error(f"Failed to read workload: {e}")
            return

        tasks = []
        for i in range(len(workload_rows)):
            model_name, time_offset, input_len, output_len = workload_rows[i]
            input_text = ' '.join(['1' for _ in range(input_len)])
            tasks.append(
                self.schedule_request(
                    model_name, input_text, output_len, time_offset
                )
            )

        await asyncio.gather(*tasks)
        self.write_latency_results()

    async def schedule_request(
        self,
        model_name: str,
        input_text: str,
        output_length: int,
        time_offset: float,
    ) -> None:
        await asyncio.sleep(time_offset)
        request_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": output_length,
        }
        await self.send_request(request_data)

    async def send_request(self, request_data: dict) -> None:
        start_time = time.time()
        try:
            response = await self.client.chat.completions.create(**request_data)
            end_time = time.time()
            latency = end_time - start_time
            model = request_data["model"]
            logger.info(
                f"Generation successful: {model} {response.id} latency={latency}"
            )
            self.latency_results.append(
                {
                    "model": model,
                    "id": response.id,
                    "latency": latency,
                    "first_token_time": response.created - start_time,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            )
            return response
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            logger.error(f"Failed to generate. Error: {str(e)}")
            self.latency_results.append(
                {
                    "model": request_data["model"],
                    "latency": latency,
                    "error": str(e),
                }
            )

    def write_latency_results(self) -> None:
        with open(self.output_path, "w") as f:
            json.dump(self.latency_results, f, indent=4)
        logger.info(f"Latency results written to {self.output_path}")
