#!/bin/bash
# move_worktree.sh <name>...  moves mstar-worktrees/<name> to mstar-worktrees/regime-batching/<name> (git worktree move) and
# repoints its uv venv: the bin/ shebangs and activate scripts, the editable finder, direct_url.json. Those are the only
# files of a 7 GB venv that hold the worktree path (full GNU grep of bench-main/.venv, 2026-10-07). It refuses a worktree
# any process still uses (cwd, exe, open file, mapped library, command line), and checks import mstar, bin/mstar and git.
# Used for 12 worktrees on 2026-10-07; bench-next and regime-harness follow once the v4 matrix (pool 13181) is done. Then
# change their WT_OLD paths in admission_matrix.sbatch to $WT.
set -euo pipefail
G=/usr/bin/grep   # the shell's grep is ugrep with --ignore-files, which skips the gitignored .venv
OLD=/shared/home/garv901-55613a/mstar-worktrees
NEW=$OLD/regime-batching
MAIN=$OLD/main
mkdir -p "$NEW"
fail() { echo "FAIL $w: $*" >&2; exit 1; }
for w in "$@"; do
    src=$OLD/$w dst=$NEW/$w
    [ -d "$src" ] || fail "$src missing"
    [ -e "$dst" ] && fail "$dst exists"
    # Nothing may run from it: cwd, exe, open files, mapped libraries or command line under $src.
    busy=
    for p in /proc/[0-9]*; do
        { readlink "$p/cwd"; readlink "$p/exe"; ls -l "$p/fd"; cat "$p/maps"; tr '\0' ' ' < "$p/cmdline"; echo; } 2>/dev/null \
            | $G -qE "$src(/|\$| )" && busy+=" ${p#/proc/}"
    done
    [ -z "$busy" ] || fail "in use by pids$busy"
    git -C "$MAIN" worktree move "$src" "$dst"
    V=$dst/.venv SP=$dst/.venv/lib/python3.12/site-packages
    if [ -d "$V" ]; then
        mapfile -t files < <($G -rlIE "$OLD/$w([^A-Za-z0-9_.-]|\$)" "$V/bin" "$SP"/__editable__* 2>/dev/null || true)
        [ ${#files[@]} -gt 0 ] && sed -i -E "s#$OLD/$w([^A-Za-z0-9_.-]|\$)#$NEW/$w\1#g" "${files[@]}"
        # direct_url.json records where the editable install came from; a cloned venv still names its source worktree.
        for du in "$SP"/mstar*.dist-info/direct_url.json; do
            [ -f "$du" ] && $G -q '"editable":true' "$du" && printf '{"url":"file://%s","dir_info":{"editable":true}}' "$dst" > "$du.tmp" && mv "$du.tmp" "$du"
        done
        rm -f "$SP"/__pycache__/__editable___*finder*.pyc
        left=$($G -rlIP "$OLD/(?!regime-batching/)" "$V/bin" "$SP"/__editable__* "$SP"/*.dist-info/direct_url.json 2>/dev/null || true)
        [ -z "$left" ] || fail "old paths left in: $left"
        got=$(cd / && "$V/bin/python" -c 'import mstar, os; print(os.path.dirname(os.path.dirname(os.path.realpath(mstar.__file__))))')
        [ "$got" = "$dst" ] || fail "import mstar resolves to $got"
        timeout 300 "$V/bin/mstar" --help > /dev/null 2>&1 || fail "bin/mstar --help failed"
        nf=${#files[@]}
    else
        nf=0
    fi
    git -C "$dst" rev-parse --show-toplevel | $G -qx "$dst" || fail "git toplevel"
    git -C "$MAIN" worktree list --porcelain | $G -qx "worktree $dst" || fail "worktree list"
    echo "OK $w -> $dst ($nf venv files repointed, branch $(git -C "$dst" rev-parse --abbrev-ref HEAD))"
done
