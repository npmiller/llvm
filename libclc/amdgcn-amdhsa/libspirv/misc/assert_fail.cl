//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

void __assert_fail(const char *assertion, const char *file, unsigned int line,
                   const char *function) {
  printf("%s:%u: %s: Device-side assertion `%s' failed.\n", file, line,
         function, assertion);
  __builtin_trap();
}
