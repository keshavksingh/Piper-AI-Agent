"""Generate Python gRPC code from all .proto files."""

import subprocess
import sys
import os

PROTO_DIR = os.path.join(os.path.dirname(__file__), "..", "protos")

PROTO_FILES = [
    "knowledge_service.proto",
    "llm_service.proto",
    "agent_service.proto",
    "memory_service.proto",
    "tool_service.proto",
    "recommendation_service.proto",
]


def main():
    for proto in PROTO_FILES:
        proto_path = os.path.join(PROTO_DIR, proto)
        if not os.path.exists(proto_path):
            print(f"SKIP: {proto} not found")
            continue

        print(f"Generating: {proto}")
        result = subprocess.run(
            [
                sys.executable, "-m", "grpc_tools.protoc",
                f"-I={PROTO_DIR}",
                f"--python_out={PROTO_DIR}",
                f"--grpc_python_out={PROTO_DIR}",
                proto_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
        else:
            print(f"  OK")

    print("\nDone!")


if __name__ == "__main__":
    main()
