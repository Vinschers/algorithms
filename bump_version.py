import toml
import sys
import subprocess

# Ensure we receive the commit message file
if len(sys.argv) < 2:
    print("❌ Error: No commit message file provided. Exiting.")
    exit(1)

commit_msg_file = sys.argv[1]

# Read the commit message
with open(commit_msg_file, "r") as f:
    commit_message = f.read().strip()

# Load pyproject.toml
with open("pyproject.toml", "r") as f:
    data = toml.load(f)

# Get current version
current_version = data["project"]["version"]

# Parse version (major.minor.patch) as integers
try:
    major, minor, patch = map(int, current_version.split("."))
except ValueError:
    print(f"❌ Error: Invalid version format '{current_version}' in pyproject.toml")
    exit(1)

# Determine bump type based on commit message
if "#major" in commit_message.lower():
    major += 1
    minor = 0
    patch = 0
    bump_type = "major"
elif "#minor" in commit_message.lower():
    minor += 1
    patch = 0
    bump_type = "minor"
elif "#patch" in commit_message.lower():
    patch += 1
    bump_type = "patch"
else:
    print("ℹ️ No version bump keyword found. Skipping.")
    exit(0)  # Exit without error if no bump is needed

# Set new version as string
new_version = f"{major}.{minor}.{patch}"
data["project"]["version"] = new_version

# Save updated pyproject.toml
with open("pyproject.toml", "w") as f:
    toml.dump(data, f)

# Stage the updated file
subprocess.run(["git", "add", "pyproject.toml"])

print(f"✅ Version bumped: {current_version} → {new_version} ({bump_type})")

print(f"✅ Version bumped: {current_version} → {new_version} ({bump_type})")
