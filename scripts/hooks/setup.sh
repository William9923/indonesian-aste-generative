#!/bin/sh

# Available git hooks
HOOK_NAMES="pre-commit"

# assuming the script is in a bin directory, one level into the repo
ROOT_DIR=$(git rev-parse --show-toplevel)
GIT_HOOK_DIR=$ROOT_DIR/.git/hooks
SCRIPT_HOOK_DIR=$ROOT_DIR/scripts/hooks

# traverse for each available hooks
for hook in $HOOK_NAMES; do
    # If the hook already exists, is executable, and is not a symlink
    if [ ! -h $GIT_HOOK_DIR/$hook -a -x $GIT_HOOK_DIR/$hook ]; then
        # move the hooks into git-hook local stage
        mv $GIT_HOOK_DIR/$hook $GIT_HOOK_DIR/$hook.local
    fi
    # create the symlink, overwriting the file if it exists
    # probably the only way this would happen is if you're using an old version of git
    # -- back when the sample hooks were not executable, instead of being named ____.sample
    ln -s -f $SCRIPT_HOOK_DIR/$hook $GIT_HOOK_DIR/$hook

    if [ -f "$SCRIPT_HOOK_DIR/$hook" ]; then
        chmod +x $GIT_HOOK_DIR/$hook
        echo "Making excutable hooks of $SCRIPT_HOOK_DIR/$hook"
    fi

done

echo "done setting up git hooks"