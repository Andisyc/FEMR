#!/usr/bin/env bash

set -o pipefail

if [[ $# -eq 0 ]]; then
    printf 'Usage: %s FILE [FILE ...]\n' "$0" >&2
    printf '   or: %s send [croc options] FILE\n' "$0" >&2
    exit 2
fi

copy_to_terminal_clipboard() {
    local value=$1

    if [[ "$(uname -s)" == "Darwin" ]] && command -v pbcopy >/dev/null 2>&1 && [[ -z "${SSH_CONNECTION:-}" ]]; then
        printf '%s' "$value" | pbcopy
        return
    fi

    # OSC 52 asks the local terminal to place this text in its clipboard.
    # This is what makes a script running on a remote server copy to the Mac.
    local encoded
    encoded=$(printf '%s' "$value" | base64 | tr -d '\r\n')
    printf '\033]52;c;%s\a' "$encoded" > /dev/tty
}

if [[ $1 == "send" ]]; then
    croc_args=("$@")
else
    croc_args=(send --no-local "$@")
fi

copied=0
status_file=$(mktemp "${TMPDIR:-/tmp}/croc-send-copy.XXXXXX")
trap 'rm -f "$status_file"' EXIT

while IFS= read -r line; do
    printf '%s\n' "$line"

    if [[ $copied -eq 0 && $line =~ (CROC_SECRET=\"[^\"]+\"[[:space:]]+croc.*)$ ]]; then
        receive_command=${BASH_REMATCH[1]}
        copy_to_terminal_clipboard "$receive_command"
        copied=1
        printf '\n[Copied to local clipboard] %s\n\n' "$receive_command"
    fi
done < <(
    croc "${croc_args[@]}" 2>&1
    printf '%s' "$?" > "$status_file"
)

status=$(cat "$status_file")
if [[ $copied -eq 0 ]]; then
    printf '\nWarning: croc did not print a receiver command; clipboard was not changed.\n' >&2
fi
exit "$status"
