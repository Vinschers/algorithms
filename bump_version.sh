#!/bin/sh

set -e # Exit on error

# Function to increment version
increment_version() {
    version="$1"
    part="$2"

    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    patch=$(echo "$version" | cut -d. -f3)

    case "$part" in
    major)
        major=$((major + 1))
        minor=0
        patch=0
        ;;
    minor)
        minor=$((minor + 1))
        patch=0
        ;;
    patch) patch=$((patch + 1)) ;;
    *)
        echo "Invalid bump type: choose major, minor, or patch" >&2
        exit 1
        ;;
    esac

    echo "$major.$minor.$patch"
}

# Get the current version from pyproject.toml
CURRENT_VERSION=$(cat ./VERSION)

# Check for argument (major, minor, or patch)
if [ -z "$1" ]; then
    echo "Usage: $0 [major|minor|patch]"
    exit 1
fi

NEW_VERSION=$(increment_version "$CURRENT_VERSION" "$1")

# Update VERSION file
echo "$NEW_VERSION" >./VERSION

# Escape dots in version numbers to avoid regex issues
ESCAPED_VERSION=$(echo "$CURRENT_VERSION" | sed 's/\./\\./g')

if [ "$(uname)" = "Darwin" ]; then
    # macOS (BSD sed)
    sed -i '' "s/$ESCAPED_VERSION/$NEW_VERSION/g" ./pyproject.toml
else
    # Linux (GNU sed)
    sed -i "s/$ESCAPED_VERSION/$NEW_VERSION/g" ./pyproject.toml
fi

# Add changes
git add ./VERSION ./pyproject.toml

echo "✅ Version bumped to $NEW_VERSION"
