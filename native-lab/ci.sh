#!/usr/bin/env bash
set -euo pipefail
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export LSAN_OPTIONS=exitcode=23
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
mkdir -p evidence
g++ --version | tee evidence/compiler.txt
uname -a | tee evidence/kernel.txt
sha256sum main.cpp test.py leak_control.cpp Makefile ci.sh | tee evidence/sources.sha256
g++ -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer leak_control.cpp -o leak-control
./leak-control >evidence/control-clean.log 2>&1
set +e
./leak-control leak >evidence/control-leak.log 2>&1
result=$?
set -e
cat evidence/control-leak.log
test "$result" = 23
grep -q 'LeakSanitizer: detected memory leaks' evidence/control-leak.log
grep -q '12345 byte(s) leaked' evidence/control-leak.log
echo 'LSAN POSITIVE AND NEGATIVE CONTROLS PASS'
make test 2>&1 | tee evidence/release-tests.log
make sanitize 2>&1 | tee evidence/sanitizer-tests.log
sha256sum glyph-native glyph-native-san | tee evidence/binaries.sha256
echo 'NATIVE LAB ASAN UBSAN LSAN PASS'
