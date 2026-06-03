# Copyright (C) 2026 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import logging
from typing import Any
from pydantic import BaseModel, Field

from rai.tools.ros2.base import BaseROS2Tool
from rai.communication.ros2.messages import ROS2Message

logger = logging.getLogger(__name__)


class DogPostureControlInput(BaseModel):
    action: str = Field(
        ...,
        description="The control instruction: 'stand_up' to make the dog stand and get ready, or 'lie_down' to make the dog lie down.",
    )


class DogPostureControlTool(BaseROS2Tool):
    name: str = "dog_posture_control"
    description: str = (
        "Control the robot dog to stand up (ready state) or lie down. "
        "This motion takes a few seconds (approx. 5-15 seconds) to complete. "
        "The tool will block until completion and automatically retry on failure."
    )
    args_schema: type[BaseModel] = DogPostureControlInput

    def _run(self, action: str) -> str:
        if action == "stand_up":
            service_name = "/nav_bridge_node/ready"
            action_desc = "stand up"
        elif action == "lie_down":
            service_name = "/nav_bridge_node/lie"
            action_desc = "lie down"
        else:
            return f"Invalid action: {action}. Must be 'stand_up' or 'lie_down'."

        max_retries = 3
        retry_delay = 2.0
        timeout_sec = 30.0

        for attempt in range(1, max_retries + 1):
            try:
                ros2_msg = ROS2Message(payload={})

                logger.info(
                    f"Attempt {attempt}/{max_retries} to call service {service_name}"
                )
                response = self.connector.service_call(
                    message=ros2_msg,
                    target=service_name,
                    msg_type="std_srvs/srv/Trigger",
                    timeout_sec=timeout_sec,
                )

                trigger_resp = response.payload
                if trigger_resp.success:
                    return f"Successfully completed {action_desc}. Dog response: {trigger_resp.message}"
                else:
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} returned failure: {trigger_resp.message}"
                    )
                    error_msg = trigger_resp.message
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed with exception: {e}"
                )
                error_msg = str(e)

            if attempt < max_retries:
                time.sleep(retry_delay)

        return f"Failed to complete {action_desc} after {max_retries} attempts. Last error: {error_msg}"
