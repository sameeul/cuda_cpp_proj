#pragma once

#include <stdio.h>
#include <string.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#else  // Linux
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#endif

inline int stringRemoveDelimiter(char delimiter, const char *string) {
  int string_start = 0;

  while (string[string_start] == delimiter) {
    string_start++;
  }

  if (string_start >= static_cast<int>(strlen(string) - 1)) {
    return 0;
  }

  return string_start;
}

inline bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref) {
  bool bFound = false;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];

      const char *equal_pos = strchr(string_argv, '=');
      int argv_length = static_cast<int>(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

      int length = static_cast<int>(strlen(string_ref));

      if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length)) {
        bFound = true;
        continue;
      }
    }
  }

  return bFound;
}

inline bool getCmdLineArgumentString(const int argc, const char **argv, const char *string_ref, char **string_retval) {
  bool bFound = false;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      char *string_argv = const_cast<char *>(&argv[i][string_start]);
      int length = static_cast<int>(strlen(string_ref));

      if (!STRNCASECMP(string_argv, string_ref, length)) {
        *string_retval = &string_argv[length + 1];
        bFound = true;
        continue;
      }
    }
  }

  if (!bFound) {
    *string_retval = NULL;
  }

  return bFound;
}
