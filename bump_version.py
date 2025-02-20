import toml
import subprocess
import os

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

# Get commit message from Git's commit edit file
commit_msg_file = ".git/COMMIT_EDITMSG"

if not os.path.exists(commit_msg_file):
    print("❌ Error: No commit message found. Exiting.")
    exit(1)

with open(commit_msg_file, "r") as f:
    commit_message = f.read().strip()

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
else:
    patch += 1
    bump_type = "patch"

# Set new version as string
new_version = f"{major}.{minor}.{patch}"
data["project"]["version"] = new_version

# Save updated pyproject.toml
with open("pyproject.toml", "w") as f:
    toml.dump(data, f)

# Stage and commit the updated version
subprocess.run(["git", "add", "pyproject.toml"])

print(f"✅ Version bumped: {current_version} → {new_version} ({bump_type})")
