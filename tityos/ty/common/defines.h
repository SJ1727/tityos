#pragma once

#if defined(_WIN32) || defined(_WIN64)
  #ifdef TITYOS_EXPORTS
    #define TITYOS_API __declspec(dllexport)
  #else
    #define TITYOS_API __declspec(dllimport)
  #endif
#else
  #define TITYOS_API __attribute__((visibility("default")))
#endif