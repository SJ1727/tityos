#pragma once

#if defined(_WIN32) || defined(_WIN64)
  #ifdef tittyos_EXPORTS
    #define tittyos_API __declspec(dllexport)
  #else
    #define tittyos_API __declspec(dllimport)
  #endif
#else
  #define tittyos_API __attribute__((visibility("default")))
#endif