#!/usr/bin/env python3
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

import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger
except ImportError:
    print("Error: ROS 2 python libraries (rclpy) not found. Please source ROS 2 first.")
    sys.exit(1)


class MockDogPostureServer(Node):
    def __init__(self):
        super().__init__("mock_dog_posture_server")

        self.ready_srv = self.create_service(
            Trigger, "/nav_bridge_node/ready", self.handle_ready
        )
        self.lie_srv = self.create_service(
            Trigger, "/nav_bridge_node/lie", self.handle_lie
        )
        self.get_logger().info("Mock Dog Posture Server is running...")
        self.get_logger().info("Services available:")
        self.get_logger().info("  - /nav_bridge_node/ready (Trigger)")
        self.get_logger().info("  - /nav_bridge_node/lie (Trigger)")

    def handle_ready(self, request, response):
        self.get_logger().info(
            "Received call to /nav_bridge_node/ready. Dog is standing up..."
        )
        time.sleep(3.0)  # Simulate mechanical action delay
        response.success = True
        response.message = "Dog stood up successfully"
        self.get_logger().info(
            "Replied: success=True, message='Dog stood up successfully'"
        )
        return response

    def handle_lie(self, request, response):
        self.get_logger().info(
            "Received call to /nav_bridge_node/lie. Dog is lying down..."
        )
        time.sleep(3.0)  # Simulate mechanical action delay
        response.success = True
        response.message = "Dog lied down successfully"
        self.get_logger().info(
            "Replied: success=True, message='Dog lied down successfully'"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    server = MockDogPostureServer()
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
