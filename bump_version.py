import toml
import argparse

# Load pyproject.toml
with open("pyproject.toml", "r") as f:
    data = toml.load(f)

# Get current version
current_version = data["project"]["version"]

# Parse current version
major, minor, patch = map(int, current_version.split("."))

# Argument parsing for version bump type
parser = argparse.ArgumentParser()
parser.add_argument("type", choices=["major", "minor", "patch"], help="Version bump type")
args = parser.parse_args()

# Increment version
if args.type == "major":
    major += 1
    minor, patch = 0, 0
elif args.type == "minor":
    minor += 1
    patch = 0
elif args.type == "patch":
    patch += 1

new_version = f"{major}.{minor}.{patch}"
print(f"Bumping version: {current_version} → {new_version}")

# Update version in pyproject.toml
data["project"]["version"] = new_version

# Save updated pyproject.toml
with open("pyproject.toml", "w") as f:
    toml.dump(data, f)

# Print new version for debugging
print(f"Updated version: {data['project']['version']}")
