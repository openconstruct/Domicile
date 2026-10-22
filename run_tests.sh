#!/usr/bin/env bash
# Domicile regression tests; --quick skips live health checks.

set -u
cd "$(dirname "$0")"

fail=0

echo "=== unit tests ==="
for t in _*_test.py; do
    [ -e "$t" ] || continue
    printf '  %-32s ' "$t"
    if out=$(PYTHONDONTWRITEBYTECODE=1 python3 _test_runner.py "$t" 2>&1); then
        echo "ok"
    else
        echo "FAIL"
        echo "$out" | sed 's/^/      /'
        fail=1
    fi
done

if [ "${1:-}" != "--quick" ]; then
    echo
    echo "=== selftest (this box, right now) ==="
    if python3 selftest.py; then
        :
    else
        echo "  (selftest reported a FAIL -- environment, not necessarily code)"
        fail=1
    fi
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "ALL GREEN"
else
    echo "SOMETHING FAILED -- look before you restart"
fi
exit "$fail"
