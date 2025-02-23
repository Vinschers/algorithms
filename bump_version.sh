#!/bin/bash

set -e  # Exit on any error

VERSION_FILE="VERSION"
PYPROJECT_FILE="pyproject.toml"

# Function to increment version
increment_version() {
    local version=$1
    local part=$2

    IFS='.' read -r major minor patch <<< "$version"

    case $part in
        major) ((major++)); minor=0; patch=0 ;;
        minor) ((minor++)); patch=0 ;;
        patch) ((patch++)) ;;
        *) echo "Invalid bump type: choose major, minor, or patch" >&2; exit 1 ;;
    esac

    echo "$major.$minor.$patch"
}

# Get the current version from pyproject.toml
CURRENT_VERSION=$(grep -E '^version\s*=' "$PYPROJECT_FILE" | sed -E 's/version\s*=\s*"([^"]+)"/\1/')
if [[ -z "$CURRENT_VERSION" ]]; then
    echo "Error: Could not find version in $PYPROJECT_FILE"
    exit 1
fi

# Check for argument (major, minor, or patch)
if [[ -z "$1" ]]; then
    echo "Usage: $0 [major|minor|patch]"
    exit 1
fi

NEW_VERSION=$(increment_version "$CURRENT_VERSION" "$1")

# Update VERSION file
echo "$NEW_VERSION" > "$VERSION_FILE"

# Update pyproject.toml version
sed -i.bak -E "s/(version\s*=\s*)\"[^\"]+\"/\1\"$NEW_VERSION\"/" "$PYPROJECT_FILE" && rm "${PYPROJECT_FILE}.bak"

# Commit changes
git add "$VERSION_FILE" "$PYPROJECT_FILE"

echo "✅ Version bumped to $NEW_VERSION"
