"""ROS2 client for running an openpi AgiBot G01 policy on the real robot.

Control is disabled by default. Always inspect model outputs with
``--no-enable-control`` before enabling command publishers.
"""

import argparse
from dataclasses import dataclass
import logging
import threading
import time

from cv_bridge import CvBridge
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

try:
    from examples.agibot_g01 import control_utils
except ModuleNotFoundError:  # Supports `python examples/agibot_g01/main.py`.
    import control_utils

try:
    from genie_msgs.msg import EndState
except ImportError as exc:  # pragma: no cover - only available in the robot ROS environment.
    raise ImportError("genie_msgs is required to read AgiBot gripper state") from exc


@dataclass(frozen=True)
class ClientConfig:
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    api_key: str | None = None
    task: str = "Fixed-point Non-generalized Door Opening"
    infer_timeout: float = 20.0
    sensor_ready_timeout: float = 30.0
    max_cycles: int = 0
    skip_metadata_check: bool = False

    enable_control: bool = False
    confirm_control: bool = True
    control_hz: float = 60.0
    model_fps: float = 30.0
    execute_horizon: int = 8
    blend_steps: int = 14
    ema_alpha: float = 1.0
    max_step_delta: float = 0.2618
    max_joint_speed: float = 3.0
    max_joint_accel: float = 6.0
    pos_tolerance: float = 0.03
    final_hold_steps: int = 5

    left_gripper_scale: float = 1.0
    left_gripper_offset: float = 0.0
    left_gripper_min: float = 0.0
    left_gripper_max: float = 0.0018111112
    right_gripper_scale: float = 1.0
    right_gripper_offset: float = 0.0
    right_gripper_min: float = 0.0
    right_gripper_max: float = 1.0

    head_camera_topic: str = "/camera/head_color"
    left_hand_camera_topic: str = "/camera/hand_left_color"
    right_hand_camera_topic: str = "/camera/hand_right_color"
    arm_state_topic: str = "/hal/arm_joint_state"
    left_ee_state_topic: str = "/hal/left_ee_data"
    right_ee_state_topic: str = "/hal/right_ee_data"
    arm_command_topic: str = "/wbc/arm_command"
    left_ee_command_topic: str = "/wbc/left_ee_command"
    right_ee_command_topic: str = "/wbc/right_ee_command"
    left_gripper_joint_name: str = "left_gripper_joint1"
    right_gripper_joint_name: str = "right_gripper_joint1"
    qos_depth: int = 1


def build_parser() -> argparse.ArgumentParser:
    defaults = ClientConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--policy-host", default=defaults.policy_host)
    parser.add_argument("--policy-port", type=int, default=defaults.policy_port)
    parser.add_argument("--api-key", default=defaults.api_key)
    parser.add_argument("--task", default=defaults.task)
    parser.add_argument("--infer-timeout", type=float, default=defaults.infer_timeout)
    parser.add_argument("--sensor-ready-timeout", type=float, default=defaults.sensor_ready_timeout)
    parser.add_argument("--max-cycles", type=int, default=defaults.max_cycles)
    parser.add_argument("--skip-metadata-check", action="store_true")

    parser.set_defaults(enable_control=defaults.enable_control, confirm_control=defaults.confirm_control)
    parser.add_argument("--enable-control", dest="enable_control", action="store_true")
    parser.add_argument("--no-enable-control", dest="enable_control", action="store_false")
    parser.add_argument("--confirm-control", dest="confirm_control", action="store_true")
    parser.add_argument("--no-confirm-control", dest="confirm_control", action="store_false")
    parser.add_argument("--control-hz", type=float, default=defaults.control_hz)
    parser.add_argument("--model-fps", type=float, default=defaults.model_fps)
    parser.add_argument("--execute-horizon", type=int, default=defaults.execute_horizon)
    parser.add_argument("--blend-steps", type=int, default=defaults.blend_steps)
    parser.add_argument("--ema-alpha", type=float, default=defaults.ema_alpha)
    parser.add_argument("--max-step-delta", type=float, default=defaults.max_step_delta)
    parser.add_argument("--max-joint-speed", type=float, default=defaults.max_joint_speed)
    parser.add_argument("--max-joint-accel", type=float, default=defaults.max_joint_accel)
    parser.add_argument("--pos-tolerance", type=float, default=defaults.pos_tolerance)
    parser.add_argument("--final-hold-steps", type=int, default=defaults.final_hold_steps)

    for side in ("left", "right"):
        parser.add_argument(f"--{side}-gripper-scale", type=float, default=getattr(defaults, f"{side}_gripper_scale"))
        parser.add_argument(f"--{side}-gripper-offset", type=float, default=getattr(defaults, f"{side}_gripper_offset"))
        parser.add_argument(f"--{side}-gripper-min", type=float, default=getattr(defaults, f"{side}_gripper_min"))
        parser.add_argument(f"--{side}-gripper-max", type=float, default=getattr(defaults, f"{side}_gripper_max"))

    for name in (
        "head_camera_topic",
        "left_hand_camera_topic",
        "right_hand_camera_topic",
        "arm_state_topic",
        "left_ee_state_topic",
        "right_ee_state_topic",
        "arm_command_topic",
        "left_ee_command_topic",
        "right_ee_command_topic",
        "left_gripper_joint_name",
        "right_gripper_joint_name",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", default=getattr(defaults, name))
    parser.add_argument("--qos-depth", type=int, default=defaults.qos_depth)
    return parser


def parse_config() -> ClientConfig:
    config = ClientConfig(**vars(build_parser().parse_args()))
    positive = {
        "policy_port": config.policy_port,
        "infer_timeout": config.infer_timeout,
        "sensor_ready_timeout": config.sensor_ready_timeout,
        "control_hz": config.control_hz,
        "model_fps": config.model_fps,
        "max_step_delta": config.max_step_delta,
        "max_joint_speed": config.max_joint_speed,
        "max_joint_accel": config.max_joint_accel,
        "pos_tolerance": config.pos_tolerance,
        "qos_depth": config.qos_depth,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if min(config.execute_horizon, config.blend_steps, config.final_hold_steps, config.max_cycles) < 0:
        raise ValueError("horizon, blend, hold, and max-cycles values must be non-negative")
    if not 0 < config.ema_alpha <= 1:
        raise ValueError("--ema-alpha must be in (0, 1]")
    if config.left_gripper_min > config.left_gripper_max:
        raise ValueError("left gripper min must be <= max")
    if config.right_gripper_min > config.right_gripper_max:
        raise ValueError("right gripper min must be <= max")
    return config


def _read_end_position(message: EndState) -> float:
    if not message.end_state:
        raise ValueError("EndState.end_state is empty")
    return float(message.end_state[0].position)


class AgiBotG01Node(Node):
    def __init__(self, config: ClientConfig):
        super().__init__("openpi_agibot_g01_client")
        self._config = config
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._images: dict[str, np.ndarray] = {}
        self._joints: np.ndarray | None = None
        self._joint_names: list[str] = []
        self._left_gripper: float | None = None
        self._right_gripper: float | None = None

        qos = config.qos_depth
        self.create_subscription(Image, config.head_camera_topic, lambda msg: self._on_image("top_head", msg), qos)
        self.create_subscription(
            Image, config.left_hand_camera_topic, lambda msg: self._on_image("hand_left", msg), qos
        )
        self.create_subscription(
            Image, config.right_hand_camera_topic, lambda msg: self._on_image("hand_right", msg), qos
        )
        self.create_subscription(JointState, config.arm_state_topic, self._on_joints, qos)
        self.create_subscription(EndState, config.left_ee_state_topic, self._on_left_gripper, qos)
        self.create_subscription(EndState, config.right_ee_state_topic, self._on_right_gripper, qos)

        self._arm_publisher = None
        self._left_publisher = None
        self._right_publisher = None
        if config.enable_control:
            self._arm_publisher = self.create_publisher(JointState, config.arm_command_topic, qos)
            self._left_publisher = self.create_publisher(JointState, config.left_ee_command_topic, qos)
            self._right_publisher = self.create_publisher(JointState, config.right_ee_command_topic, qos)

    def _on_image(self, key: str, message: Image) -> None:
        image = np.asarray(self._bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"), dtype=np.uint8)
        with self._lock:
            self._images[key] = image.copy()

    def _on_joints(self, message: JointState) -> None:
        if len(message.position) < 14:
            self.get_logger().error(f"Ignoring arm state with {len(message.position)} positions; expected at least 14")
            return
        positions = np.asarray(message.position[:14], dtype=np.float32)
        if not np.all(np.isfinite(positions)):
            self.get_logger().error("Ignoring non-finite arm state")
            return
        with self._lock:
            self._joints = positions
            self._joint_names = list(message.name[:14])

    def _on_left_gripper(self, message: EndState) -> None:
        try:
            position = _read_end_position(message)
        except (ValueError, TypeError, IndexError) as exc:
            self.get_logger().error(f"Invalid left gripper state: {exc}")
            return
        with self._lock:
            self._left_gripper = position

    def _on_right_gripper(self, message: EndState) -> None:
        try:
            position = _read_end_position(message)
        except (ValueError, TypeError, IndexError) as exc:
            self.get_logger().error(f"Invalid right gripper state: {exc}")
            return
        with self._lock:
            self._right_gripper = position

    def ready(self) -> bool:
        with self._lock:
            return (
                set(self._images) == {"top_head", "hand_left", "hand_right"}
                and self._joints is not None
                and self._left_gripper is not None
                and self._right_gripper is not None
            )

    def snapshot(self, prompt: str) -> tuple[dict, np.ndarray]:
        with self._lock:
            if not self.ready_unlocked():
                raise RuntimeError("Robot sensors are not ready")
            joints = self._joints.copy()
            state = np.concatenate(
                (joints, np.asarray([self._left_gripper, self._right_gripper], dtype=np.float32))
            )
            observation = {
                "images": {key: value.copy() for key, value in self._images.items()},
                "state": state,
                "prompt": prompt,
            }
        return observation, joints

    def ready_unlocked(self) -> bool:
        return (
            set(self._images) == {"top_head", "hand_left", "hand_right"}
            and self._joints is not None
            and self._left_gripper is not None
            and self._right_gripper is not None
        )

    def current_joints(self) -> np.ndarray:
        with self._lock:
            if self._joints is None:
                raise RuntimeError("Arm joint state is unavailable")
            return self._joints.copy()

    def publish(self, joints: np.ndarray, left_gripper: float, right_gripper: float) -> None:
        if not self._config.enable_control:
            return
        if not np.all(np.isfinite(joints)) or not np.isfinite(left_gripper) or not np.isfinite(right_gripper):
            raise ValueError("Refusing to publish non-finite command")
        stamp = self.get_clock().now().to_msg()
        arm = JointState()
        arm.header.stamp = stamp
        arm.name = self._joint_names if len(self._joint_names) == 14 else [f"arm_joint_{i + 1}" for i in range(14)]
        arm.position = np.asarray(joints, dtype=np.float64).tolist()
        self._arm_publisher.publish(arm)
        self._publish_gripper(self._left_publisher, self._config.left_gripper_joint_name, left_gripper, stamp)
        self._publish_gripper(self._right_publisher, self._config.right_gripper_joint_name, right_gripper, stamp)

    @staticmethod
    def _publish_gripper(publisher, joint_name: str, position: float, stamp) -> None:
        command = JointState()
        command.header.stamp = stamp
        command.name = [joint_name]
        command.position = [float(position)]
        publisher.publish(command)


def _scale_gripper(value: float, *, scale: float, offset: float, minimum: float, maximum: float) -> float:
    return float(np.clip(value * scale + offset, minimum, maximum))


def wait_for_sensors(node: AgiBotG01Node, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while rclpy.ok() and not node.ready():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Robot sensors did not become ready within {timeout:.1f}s")
        time.sleep(0.05)


def execute_chunk(
    node: AgiBotG01Node,
    config: ClientConfig,
    actions: np.ndarray,
    limiter: control_utils.JointSafetyLimiter,
    ema: control_utils.EMAFilter,
) -> None:
    chunk = control_utils.prepare_execution_chunk(
        actions,
        execute_horizon=config.execute_horizon,
        model_fps=config.model_fps,
        control_hz=config.control_hz,
    )
    initial_joints = node.current_joints()
    limiter.reset(initial_joints)
    ema.reset(initial_joints)
    period = 1.0 / config.control_hz
    next_tick = time.monotonic()
    last_command = initial_joints
    left = right = 0.0

    for step, action in enumerate(chunk):
        if not rclpy.ok():
            return
        target = control_utils.blend_target(initial_joints, action[:14], step, config.blend_steps)
        last_command = ema(limiter(target))
        left = _scale_gripper(
            action[14],
            scale=config.left_gripper_scale,
            offset=config.left_gripper_offset,
            minimum=config.left_gripper_min,
            maximum=config.left_gripper_max,
        )
        right = _scale_gripper(
            action[15],
            scale=config.right_gripper_scale,
            offset=config.right_gripper_offset,
            minimum=config.right_gripper_min,
            maximum=config.right_gripper_max,
        )
        node.publish(last_command, left, right)
        next_tick += period
        time.sleep(max(0.0, next_tick - time.monotonic()))

    for _ in range(config.final_hold_steps):
        node.publish(last_command, left, right)
        time.sleep(period)
    error = float(np.max(np.abs(node.current_joints() - last_command)))
    if error > config.pos_tolerance:
        node.get_logger().warning(
            f"Final joint tracking error {error:.4f} rad exceeds tolerance {config.pos_tolerance:.4f} rad"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = parse_config()
    if config.enable_control and config.confirm_control:
        answer = input("Real-robot control is enabled. Type ENABLE to continue: ").strip()
        if answer != "ENABLE":
            raise RuntimeError("Control activation was not confirmed")

    rclpy.init()
    node = AgiBotG01Node(config)
    ros_executor = MultiThreadedExecutor(num_threads=4)
    ros_executor.add_node(node)
    ros_thread = threading.Thread(target=ros_executor.spin, name="agibot-ros-executor", daemon=True)
    ros_thread.start()

    try:
        wait_for_sensors(node, config.sensor_ready_timeout)
        policy = WebsocketClientPolicy(
            host=config.policy_host,
            port=config.policy_port,
            api_key=config.api_key,
            receive_timeout=config.infer_timeout,
        )
        metadata = policy.get_server_metadata()
        if not config.skip_metadata_check:
            control_utils.validate_server_metadata(metadata, expected_horizon=16, expected_fps=int(config.model_fps))
        control_status = "ENABLED" if config.enable_control else "disabled"
        node.get_logger().info(f"Connected to policy server; control={control_status}; metadata={metadata}")

        limiter = control_utils.JointSafetyLimiter(
            dim=14,
            dt=1.0 / config.control_hz,
            max_step_delta=config.max_step_delta,
            max_speed=config.max_joint_speed,
            max_accel=config.max_joint_accel,
        )
        ema = control_utils.EMAFilter(14, config.ema_alpha)
        cycle = 0
        while rclpy.ok() and (config.max_cycles <= 0 or cycle < config.max_cycles):
            observation, _ = node.snapshot(config.task)
            response = policy.infer(observation)
            actions = control_utils.parse_action_chunk(response)
            cycle += 1
            node.get_logger().info(
                f"cycle={cycle} actions={actions.shape} joints=[{actions[:, :14].min():.3f}, "
                f"{actions[:, :14].max():.3f}] grippers=[{actions[:, 14].min():.6f}, "
                f"{actions[:, 14].max():.6f}]/[{actions[:, 15].min():.3f}, {actions[:, 15].max():.3f}]"
            )
            if config.enable_control:
                execute_chunk(node, config, actions, limiter, ema)
    finally:
        ros_executor.shutdown(timeout_sec=2.0)
        ros_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
