#!/usr/bin/env python3
"""Publish Task Board ROS 2 messages for testing the Isaac Sim digital twin.

Run this from a normal ROS 2 Jazzy terminal, not from Isaac Sim:

    source /opt/ros/jazzy/setup.bash
    python3 taskboard_ros_publisher.py --interactive

Examples:
    python3 taskboard_ros_publisher.py --red 1 --blue 0 --slider 0.8 --door 1.0
    python3 taskboard_ros_publisher.py --pulse-red
    python3 taskboard_ros_publisher.py --interactive
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


TOPIC_PREFIX = "/task_board"


class TaskboardPublisher(Node):
    def __init__(self):
        super().__init__("taskboard_test_publisher")
        self.red_pub = self.create_publisher(Bool, f"{TOPIC_PREFIX}/red_button", 10)
        self.blue_pub = self.create_publisher(Bool, f"{TOPIC_PREFIX}/blue_button", 10)
        self.slider_pub = self.create_publisher(Float32, f"{TOPIC_PREFIX}/slider", 10)
        self.door_pub = self.create_publisher(Float32, f"{TOPIC_PREFIX}/door", 10)

    def publish_button(self, name, value):
        msg = Bool()
        msg.data = bool(value)
        if name == "red":
            self.red_pub.publish(msg)
            topic = f"{TOPIC_PREFIX}/red_button"
        elif name == "blue":
            self.blue_pub.publish(msg)
            topic = f"{TOPIC_PREFIX}/blue_button"
        else:
            raise ValueError(f"unknown button: {name}")
        self.get_logger().info(f"published {topic}: {msg.data}")

    def publish_slider(self, value):
        msg = Float32()
        msg.data = clamp01(value)
        self.slider_pub.publish(msg)
        self.get_logger().info(f"published {TOPIC_PREFIX}/slider: {msg.data:.3f}")

    def publish_door(self, value):
        msg = Float32()
        msg.data = clamp01(value)
        self.door_pub.publish(msg)
        self.get_logger().info(f"published {TOPIC_PREFIX}/door: {msg.data:.3f}")

    def pulse_button(self, name, seconds=0.25):
        self.publish_button(name, True)
        spin_briefly(self, seconds)
        self.publish_button(name, False)


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def spin_briefly(node, seconds=0.1):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)


def parse_bool(text):
    normalized = str(text).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on", "pressed"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", "released"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {text!r}")


def interactive_loop(node):
    print("Interactive Task Board publisher")
    print("Commands:")
    print("  r 1|0       red button pressed/released")
    print("  b 1|0       blue button pressed/released")
    print("  pr          pulse red button")
    print("  pb          pulse blue button")
    print("  s VALUE     slider 0.0..1.0")
    print("  d VALUE     door 0.0 closed .. 1.0 open")
    print("  reset       red=0 blue=0 slider=0.5 door=0.0")
    print("  q           quit")

    while rclpy.ok():
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue
        parts = line.split()
        command = parts[0].lower()

        try:
            if command in {"q", "quit", "exit"}:
                return
            if command == "r" and len(parts) == 2:
                node.publish_button("red", parse_bool(parts[1]))
            elif command == "b" and len(parts) == 2:
                node.publish_button("blue", parse_bool(parts[1]))
            elif command == "pr" and len(parts) == 1:
                node.pulse_button("red")
            elif command == "pb" and len(parts) == 1:
                node.pulse_button("blue")
            elif command == "s" and len(parts) == 2:
                node.publish_slider(float(parts[1]))
            elif command == "d" and len(parts) == 2:
                node.publish_door(float(parts[1]))
            elif command == "reset" and len(parts) == 1:
                node.publish_button("red", False)
                node.publish_button("blue", False)
                node.publish_slider(0.5)
                node.publish_door(0.0)
            else:
                print("unknown command")
                continue
        except (ValueError, argparse.ArgumentTypeError) as exc:
            print(f"invalid value: {exc}")
            continue

        spin_briefly(node)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Publish /task_board ROS 2 messages for the Isaac Sim digital twin."
    )
    parser.add_argument("--red", type=parse_bool, help="publish red button state")
    parser.add_argument("--blue", type=parse_bool, help="publish blue button state")
    parser.add_argument("--slider", type=float, help="publish slider value, clamped to 0.0..1.0")
    parser.add_argument("--door", type=float, help="publish door value, clamped to 0.0..1.0")
    parser.add_argument("--pulse-red", action="store_true", help="publish red true then false")
    parser.add_argument("--pulse-blue", action="store_true", help="publish blue true then false")
    parser.add_argument("--interactive", "-i", action="store_true", help="start an interactive command prompt")
    parser.add_argument("--hold", type=float, default=0.25, help="seconds to hold pulse buttons")
    return parser


def main():
    args = build_parser().parse_args()

    rclpy.init()
    node = TaskboardPublisher()
    try:
        # Give DDS discovery a short moment so one-shot publishes are visible.
        spin_briefly(node, 0.5)

        if args.interactive:
            interactive_loop(node)
            return

        published = False
        if args.red is not None:
            node.publish_button("red", args.red)
            published = True
        if args.blue is not None:
            node.publish_button("blue", args.blue)
            published = True
        if args.slider is not None:
            node.publish_slider(args.slider)
            published = True
        if args.door is not None:
            node.publish_door(args.door)
            published = True
        if args.pulse_red:
            node.pulse_button("red", args.hold)
            published = True
        if args.pulse_blue:
            node.pulse_button("blue", args.hold)
            published = True

        if not published:
            print("No messages requested. Use --interactive or pass --red/--blue/--slider/--door.")

        spin_briefly(node, 0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
